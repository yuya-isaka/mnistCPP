#include "mnist.h"
#include "rwfile.h"
#include <functional>
#include <list>
#include <memory>
#include <random>

class AbstractLayer {
public:
	virtual void reset(Matrix const &t) = 0;
	virtual Matrix forward(Matrix const &in) = 0;
	virtual Matrix backward(Matrix const &out) = 0;
	virtual void learn(Matrix::real_t learning_rate) { (void)learning_rate; }
};

class AffineLayer : public AbstractLayer {
private:
	Matrix X;
	Matrix dW, dB;
public:
	Matrix W, B;
	AffineLayer() = default;
	AffineLayer(int input, int output, std::function<Matrix::real_t()> const &rand)
	{
		make(input, output);

		for (size_t i = 0; i < W.size(); i++) {
			W.data()[i] = rand();
		}
	}
	void make(int input, int output)
	{
		W.make(input, output);
		B.make(1, output);
	}
	void reset(Matrix const &t)
	{
		(void)t;
		X = {};
		dW = {};
		dB = {};
	}
	Matrix forward(Matrix const &in)
	{
		X = in;
		return in.dot(W).add(B);
	}
	Matrix backward(Matrix const &out)
	{
		Matrix dx = out.dot(W.transpose());
		dW = X.transpose().dot(out);
		dB = out.sum();
		return dx;
	}
	void learn(Matrix::real_t learning_rate)
	{
		W = W.sub(dW.mul(learning_rate));
		B = B.sub(dB.mul(learning_rate));
	}
};

class SigmoidLayer : public AbstractLayer {
private:
	Matrix Y;
public:
	void reset(Matrix const &t)
	{
		(void)t;
		Y = {};
	}
	Matrix forward(Matrix const &in)
	{
		Y = in.sigmoid();
		return Y;
	}
	Matrix backward(Matrix const &out)
	{
		if (Y.rows() != out.rows()) return {};
		if (Y.cols() != out.cols()) return {};
		Matrix dx;
		dx.make(Y.rows(), Y.cols());
		size_t n = Y.size();
		for (size_t i = 0; i < n; i++) {
			dx.data()[i] = (1 - Y.data()[i]) * Y.data()[i] * out.data()[i];
		}
		return dx;
	}
};

class SoftmaxLayer : public AbstractLayer {
private:
	Matrix T;
	Matrix Y;
public:
	void reset(Matrix const &t)
	{
		T = t;
		Y = {};
	}
	Matrix forward(Matrix const &in)
	{
		Y = in.softmax();
//		Matrix loss = Y.cross_entropy_error(T);
		return Y;
	}
	Matrix backward(Matrix const &out)
	{
		return out.sub(T).div(out.rows());
	}
};

class Random {
private:
	typedef Matrix::real_t T;
	std::random_device seed_gen;
	std::default_random_engine engine;
	std::normal_distribution<T> dist;
public:
	Random()
		: engine(seed_gen())
		, dist(0.0, 0.1)
	{
	}
	T next()
	{
		return dist(engine);
	}
};


class TwoLayerNet {
public:
	std::vector<std::shared_ptr<AbstractLayer>> layers;

	void addAffineLayer(int input, int output, std::function<Matrix::real_t()> const &rand)
	{
		layers.push_back(std::shared_ptr<AbstractLayer>(new AffineLayer(input, output, rand)));
	}

	void addSigmoidLayer()
	{
		layers.push_back(std::shared_ptr<AbstractLayer>(new SigmoidLayer()));
	}

	void addSoftmaxLayer()
	{
		layers.push_back(std::shared_ptr<AbstractLayer>(new SoftmaxLayer()));
	}

	TwoLayerNet()
	{
		int input = 28 * 28;
		int hidden1 = 50;
		int hidden2 = 50;
		int hidden3 = 50;
		int output = 10;

		Random random;
		auto Rand = [&](){
			return random.next();
		};

		addAffineLayer(input, hidden1, Rand);
		addSigmoidLayer();
		addAffineLayer(hidden1, hidden2, Rand);
		addSigmoidLayer();
		addAffineLayer(hidden2, hidden3, Rand);
		addSigmoidLayer();
		addAffineLayer(hidden3, output, Rand);
		addSoftmaxLayer();
	}

	Matrix predict(Matrix const &x)
	{
		Matrix y = x;
		for (auto &p : layers) {
			y = p->forward(y);
		}
		return y;
	}

	Matrix::real_t accuracy(Matrix const &x, Matrix const &t)
	{
		auto argmax = [](Matrix const &a, int row){
			int i = 0;
			for (size_t j = 1; j < a.cols(); j++) {
				if (a.at(row, j) > a.at(row, i)) {
					i = j;
				}
			}
			return i;
		};

		int rows = std::min(x.rows(), t.rows());
		Matrix y = predict(x);
		int acc = 0;
		for (int row = 0; row < rows; row++) {
			auto a = argmax(y, row);
			auto b = argmax(t, row);
			if (a == b) {
				acc++;
			}
		}
		return (Matrix::real_t)acc / rows;
	}

	void gradient(Matrix const &x, Matrix const &t)
	{
		for (auto &p : layers) {
			p->reset(t);
		}
		Matrix y = x;
		for (auto &p : layers) {
			y = p->forward(y);
		}
		for (auto it = layers.rbegin(); it != layers.rend(); it++) {
			auto &p = *it;
			y = p->backward(y);
		}
	}

	void train(Matrix const &x_batch, Matrix const &t_batch, double learning_rate)
	{
		gradient(x_batch, t_batch);
		for (auto &p : layers) {
			p->learn(learning_rate);
		}
	}
};

int main()
{
	mnist::DataSet train;
	if (!train.load("train-labels-idx1-ubyte", "train-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	mnist::DataSet t10k;
	if (!t10k.load("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	int iteration = 1000;
	int batch_size = 100;
	Matrix::real_t learning_rate = 0.2;

	TwoLayerNet net;

	unsigned int k = 0;
	for (int i = 0; i < iteration; i++) {
		Matrix x_batch;
		Matrix t_batch;
		for (int j = 0; j < batch_size; j++) {
			Matrix x, t;
			k = (k + rand()) % train.size();
			train.image_to_matrix(k, &x);
			train.label_to_matrix(k, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}

		net.train(x_batch, t_batch, learning_rate);

		if ((i + 1) % 100 == 0) {
			Matrix::real_t t = net.accuracy(x_batch, t_batch);
			printf("[train %d] %f\n", i + 1, t);
		}
	}

	{
		Matrix x_batch;
		Matrix t_batch;
		for (size_t j = 0; j < t10k.size(); j++) {
			Matrix x, t;
			t10k.image_to_matrix(j, &x);
			t10k.label_to_matrix(j, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}
		Matrix::real_t t = net.accuracy(x_batch, t_batch);
		printf("[t10k] %f\n", t);
	}

	return 0;
}