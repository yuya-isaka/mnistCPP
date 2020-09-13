#include "rwfile.h"
#include <fcntl.h>
#include <sys/stat.h>

bool readfile(const char *path, std::vector<char> *out)
{
    bool ok = false;
    out->clear(); // 配列を綺麗にする
    int fd = open(path, O_RDONLY);
    if (fd != -1) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size >0) {
            out->resize(st.st_size);
            if (read(fd, out->data(), out->size()) == st.st_size) {
                ok = true;
            }
        }
        close(fd);
    }
    return ok;
}