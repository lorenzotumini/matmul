#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#endif

#define NOBDEF static inline
#define NOB_IMPLEMENTATION
#include "deps/nob/nob.h"

/* Nob owns the workflow; CMake owns sources, flags and dependencies. */
static void usage(void) {
    puts("Usage: nob [build|test|bench|profile] [options] [-- tool arguments]\n"
         "  build       Compile (default command, debug mode)\n"
         "  test        Compile and run CTest (default: debug)\n"
         "  bench       Compile and run the benchmark (default: release)\n"
         "  profile     Compile and run a profiling workload (default: profile)\n"
         "Options:\n"
         "  --cc NAME      gcc (default), clang, msvc, clang-cl\n"
         "  --mode NAME    debug, release, profile (optimized with symbols)\n"
         "  --filter REGEX Run matching CTest names (test only)\n"
         "  -j, --jobs N   Parallel build jobs (default: build tool's choice)\n"
         "  -h, --help     Show help\n"
         "Examples:\n"
         "  nob test\n"
         "  nob test --filter kernels -- -V\n"
         "  nob bench -- --impl naive --m 256 --n 256 --k 256\n"
         "  nob profile --cc clang-cl -- --impl naive --repeats 200\n"
         "Run from the repo root. Requires CMake and Ninja.\n"
         "MSVC/clang-cl require a Visual Studio Developer terminal on Windows.");
}

static bool positive_count(const char *s) {
    size_t count = 0;
    if (!*s) return false;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return false;
        size_t digit = (size_t)(*s - '0');
        if (count > (SIZE_MAX - digit) / 10) return false;
        count = count * 10 + digit;
    }
    return count != 0;
}

/* Keep configure/build chatter off stdout so `nob bench -- --csv > file`
 * produces clean CSV. Children inherit stderr as their stdout during setup.
 */
static bool run_setup(Nob_Cmd *cmd) {
    fflush(stdout);
#ifdef _WIN32
    HANDLE saved = GetStdHandle(STD_OUTPUT_HANDLE);
    if (saved == INVALID_HANDLE_VALUE ||
        !SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE))) {
        nob_log(NOB_ERROR, "Could not redirect build output to stderr");
        return false;
    }
    bool ok = nob_cmd_run(cmd);
    if (!SetStdHandle(STD_OUTPUT_HANDLE, saved)) return false;
#else
    int saved = dup(STDOUT_FILENO);
    if (saved < 0) return false;
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        close(saved);
        return false;
    }
    bool ok = nob_cmd_run(cmd);
    if (dup2(saved, STDOUT_FILENO) < 0) ok = false;
    close(saved);
#endif
    return ok;
}

int main(int argc, char **argv) {
    if (nob_file_exists("nob.c") != 1 || nob_file_exists("CMakeLists.txt") != 1) {
        nob_log(NOB_ERROR, "Run nob from the matmul repository root");
        return 1;
    }
    NOB_GO_REBUILD_URSELF_PLUS(argc, argv, "deps/nob/nob.h");

    const char *action = NULL, *cc = "gcc", *mode = NULL;
    const char *filter = NULL, *jobs = NULL;
    int forwarded = argc;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--") == 0) { forwarded = i + 1; break; }
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            usage();
            return 0;
        }
        const char **value = NULL;
        if (strcmp(arg, "--cc") == 0) value = &cc;
        else if (strcmp(arg, "--mode") == 0) value = &mode;
        else if (strcmp(arg, "--filter") == 0) value = &filter;
        else if (strcmp(arg, "--jobs") == 0 || strcmp(arg, "-j") == 0) value = &jobs;
        if (value) {
            if (++i >= argc || !argv[i][0]) {
                nob_log(NOB_ERROR, "%s requires a value", arg);
                return 1;
            }
            *value = argv[i];
        } else if (!action && arg[0] != '-') {
            action = arg;
        } else {
            nob_log(NOB_ERROR, "Unknown argument: %s. Use --help; put tool arguments after --", arg);
            return 1;
        }
    }
    if (!action) action = "build";
    if (strcmp(action, "benchmark") == 0) action = "bench";
    bool building = strcmp(action, "build") == 0;
    bool testing = strcmp(action, "test") == 0;
    bool benchmarking = strcmp(action, "bench") == 0;
    bool profiling = strcmp(action, "profile") == 0;
    if (!building && !testing && !benchmarking && !profiling) {
        nob_log(NOB_ERROR, "Unknown command: %s", action);
        return 1;
    }
    if (!mode) mode = profiling ? "profile" : benchmarking ? "release" : "debug";
    const char *configuration = NULL;
    if (strcmp(mode, "debug") == 0) configuration = "Debug";
    else if (strcmp(mode, "release") == 0) configuration = "Release";
    else if (strcmp(mode, "profile") == 0) configuration = "RelWithDebInfo";
    if (!configuration || (strcmp(cc, "gcc") != 0 && strcmp(cc, "clang") != 0 &&
        strcmp(cc, "msvc") != 0 && strcmp(cc, "clang-cl") != 0) ||
        (filter && !testing) || (jobs && !positive_count(jobs)) ||
        (building && forwarded < argc)) {
        nob_log(NOB_ERROR, "Invalid compiler, mode or options for %s. Use --help", action);
        return 1;
    }

    const char *directory = nob_temp_sprintf("build/%s/%s", cc, mode);
    const char *compiler = strcmp(cc, "msvc") == 0 ? "cl" : cc;
    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "cmake", "-S", ".", "-B", directory, "-G", "Ninja",
                   nob_temp_sprintf("-DCMAKE_C_COMPILER=%s", compiler),
                   nob_temp_sprintf("-DCMAKE_BUILD_TYPE=%s", configuration),
                   "-DBUILD_TESTING=ON");
    if (!run_setup(&cmd)) return 1;
    nob_cmd_append(&cmd, "cmake", "--build", directory, "--parallel");
    if (jobs) nob_cmd_append(&cmd, jobs);
    if (!run_setup(&cmd)) return 1;
    if (building) return 0;

    if (testing) {
        nob_cmd_append(&cmd, "ctest", "--test-dir", directory,
                       "--output-on-failure", "--no-tests=error");
        if (filter) nob_cmd_append(&cmd, "-R", filter);
    } else {
#ifdef _WIN32
        const char *suffix = ".exe";
#else
        const char *suffix = "";
#endif
        nob_cmd_append(&cmd, nob_temp_sprintf("./%s/matmul_bench%s", directory, suffix));
        if (profiling) nob_cmd_append(&cmd, "--profile");
    }
    for (int i = forwarded; i < argc; ++i) nob_cmd_append(&cmd, argv[i]);
    bool ok = nob_cmd_run(&cmd);
    nob_cmd_free(cmd);
    return ok ? 0 : 1;
}
