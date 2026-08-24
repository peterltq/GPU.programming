#ifndef HELPER_STRING_H
#define HELPER_STRING_H

#include <string.h>
#include <stdlib.h>

inline bool checkCmdLineFlag(int argc, const char **argv, const char *string_ref)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!argv[i])
            continue;
        if (!strcmp(argv[i], string_ref))
            return true;
    }
    return false;
}

inline int getCmdLineArgumentInt(int argc, const char **argv, const char *string_ref)
{
    int value = -1;
    for (int i = 1; i < argc; ++i)
    {
        if (!argv[i])
            continue;
        if (!strcmp(argv[i], string_ref))
        {
            if (++i < argc)
                value = atoi(argv[i]);
        }
    }
    return value;
}

#endif // HELPER_STRING_H