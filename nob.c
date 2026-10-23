#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#endif

#define NOBDEF static inline
#define NOB_IMPLEMENTATION
#include "deps/nob.h"

static bool verbose = false;
static const char *log_directory = NULL;

/* Nob owns the workflow; CMake owns sources, flags and dependencies. */
static void usage(void) {
    puts("Usage: nob [build|test|bench|profile] [options] [-- tool arguments]\n"
         "  build       Compile (default command, debug mode)\n"
         "  test        Compile and run CTest (default: debug)\n"
         "  bench       Compile and run the benchmark (default: release)\n"
         "  profile     Build, test, and capture a Samply CPU profile\n"

         "\nOptions:\n"
         "  --cc NAME      gcc, clang, msvc, clang-cl (default: gcc;\n"
         "                 profile on Windows defaults to clang-cl for PDBs)\n"
         "  --mode NAME    debug, release, profile (optimized with symbols)\n"
         "  --filter REGEX Run matching CTest names (test only)\n"
         "  -j, --jobs N   Parallel build jobs (default: build tool's choice)\n"
         "  --output FILE  Capture path (profile only; default: unique results/ file)\n"
         "  --open         Open Samply's interactive viewer after capture\n"
         "  --verbose      Show build commands and full setup/profiler output\n"
         "  -h, --help     Show help\n"

         "\nExamples:\n"
         "  nob test\n"
         "  nob test --filter kernels -- -V\n"
         "  nob bench -- --impl naive --m 256 --n 256 --k 256\n"
         "  nob profile -- --impl naive\n"
         "  nob profile --open -- --impl naive --repeats 200\n"

         "\nRun from the repo root. Requires CMake and Ninja.\n"
         "Profile requires Samply; defaults to naive, 200 measured calls.\n"
         "If Windows compiler discovery fails, use an x64 VS Developer terminal.");
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

static void show_log(const char *path) {
    Nob_String_Builder contents = {0};
    if (nob_file_exists(path) == 1 && nob_read_entire_file(path, &contents) && contents.count) {
        fprintf(stderr, "\n--- %s ---\n", path);
        fwrite(contents.items, 1, contents.count, stderr);
        if (contents.items[contents.count - 1] != '\n') fputc('\n', stderr);
    }
    nob_sb_free(contents);
}

/* Results stay on stdout; setup details are saved and shown on failure or
 * with --verbose. Separate files avoid mixing stdout/stderr file offsets.
 */
static bool run_logged(Nob_Cmd *cmd, const char *prefix) {
    const char *out = nob_temp_sprintf("%s.stdout.log", prefix);
    const char *err = nob_temp_sprintf("%s.stderr.log", prefix);
    bool ok = nob_cmd_run(cmd, .stdout_path = out, .stderr_path = err);
    if (!ok || verbose) {
        show_log(out);
        show_log(err);
    }
    if (!ok) fprintf(stderr, "Logs: %s and %s\n", out, err);
    return ok;
}

static bool run_setup(Nob_Cmd *cmd, const char *stage) {
    return run_logged(cmd, nob_temp_sprintf("%s/%s", log_directory, stage));
}

static const char *capture_path(void) {
    time_t now = time(NULL);
    struct tm *calendar = localtime(&now);
    char stamp[32];
    if (!calendar || !strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", calendar))
        return NULL;
    return nob_temp_sprintf("results/profile-%s-%llu.json.gz", stamp,
                            (unsigned long long)nob_nanos_since_unspecified_epoch());
}

static bool capture(Nob_Cmd *workload, const char *output, bool open_viewer) {
    if (!nob_mkdir_if_not_exists("results")) return false;

    /* Check arguments/output before starting the expensive ETW/perf capture.
     * This is outside the trace. Tests have already checked correctness.
     */
    Nob_Cmd check = {0};
    nob_cmd_extend(&check, workload);
    nob_cmd_append(&check, "--validate-only");
    bool ok = run_setup(&check, "validation");
    nob_cmd_free(check);
    if (!ok) return false;
    fprintf(stderr, "[check] Requested workload passed.\n");

    Nob_Cmd recorder = {0};
    nob_cmd_append(&recorder, "samply", "record", "--unstable-presymbolicate");
    if (!open_viewer) nob_cmd_append(&recorder, "--save-only");
    nob_cmd_append(&recorder, "-o", output, "--");
    nob_cmd_extend(&recorder, workload);

    /* Save the exact command beside the profile, without shell interpretation. */
    Nob_String_Builder note = {0};
    nob_sb_append_cstr(&note, "Working directory: ");
    nob_sb_append_cstr(&note, nob_get_current_dir_temp());
    nob_sb_append_cstr(&note, "\nCapture command: ");
    nob_cmd_render(recorder, &note);
    nob_sb_append_cstr(&note, "\n");
    ok = nob_write_entire_file(nob_temp_sprintf("%s.command.txt", output),
                               note.items, note.count);
    nob_sb_free(note);
    if (ok) {
        fprintf(stderr, "[profile] Recording CPU samples...\n");
        if (open_viewer) {
            fprintf(stderr, "[profile] Viewer opens after recording; Ctrl+C stops its server.\n");
            ok = nob_cmd_run(&recorder);
        } else {
            ok = run_logged(&recorder, output);
        }
    }
    nob_cmd_free(recorder);
    if (!ok) {
        nob_log(NOB_ERROR, "Samply capture failed. Check its error above; Windows ETW may require an elevated terminal");
        return false;
    }
    if (nob_file_exists(output) != 1) {
        nob_log(NOB_ERROR, "Samply did not produce %s", output);
        return false;
    }
    fprintf(stderr, "\nSaved profile: %s\n", output);
    fprintf(stderr, "View later: samply load \"%s\"\n", output);
    return true;
}

int main(int argc, char **argv) {
    /* Configure logging before Nob's automatic rebuild too. */
    for (int i = 1; i < argc && strcmp(argv[i], "--") != 0; ++i)
        if (strcmp(argv[i], "--verbose") == 0) verbose = true;
    nob_minimal_log_level = verbose ? NOB_INFO : NOB_WARNING;
    if (nob_file_exists("nob.c") != 1 || nob_file_exists("CMakeLists.txt") != 1) {
        nob_log(NOB_ERROR, "Run nob from the matmul repository root");
        return 1;
    }
    NOB_GO_REBUILD_URSELF_PLUS(argc, argv, "deps/nob.h");

    const char *action = NULL, *cc = NULL, *mode = NULL;
    const char *filter = NULL, *jobs = NULL, *output = NULL;
    bool open_viewer = false;
    int forwarded = argc;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--") == 0) { forwarded = i + 1; break; }
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            usage();
            return 0;
        }
        if (strcmp(arg, "--open") == 0) { open_viewer = true; continue; }
        if (strcmp(arg, "--verbose") == 0) continue;
        const char **value = NULL;
        if (strcmp(arg, "--cc") == 0) value = &cc;
        else if (strcmp(arg, "--mode") == 0) value = &mode;
        else if (strcmp(arg, "--filter") == 0) value = &filter;
        else if (strcmp(arg, "--output") == 0) value = &output;
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
    bool building = strcmp(action, "build") == 0;
    bool testing = strcmp(action, "test") == 0;
    bool benchmarking = strcmp(action, "bench") == 0;
    bool profiling = strcmp(action, "profile") == 0;
    if (!building && !testing && !benchmarking && !profiling) {
        nob_log(NOB_ERROR, "Unknown command: %s", action);
        return 1;
    }
    if (!mode) mode = profiling ? "profile" : benchmarking ? "release" : "debug";
    if (!cc) {
#ifdef _WIN32
        cc = profiling ? "clang-cl" : "gcc";
#else
        cc = "gcc";
#endif
    }
    const char *configuration = NULL;
    if (strcmp(mode, "debug") == 0) configuration = "Debug";
    else if (strcmp(mode, "release") == 0) configuration = "Release";
    else if (strcmp(mode, "profile") == 0) configuration = "RelWithDebInfo";
    if (!configuration || (strcmp(cc, "gcc") != 0 && strcmp(cc, "clang") != 0 &&
        strcmp(cc, "msvc") != 0 && strcmp(cc, "clang-cl") != 0) ||
        (filter && !testing) || (jobs && !positive_count(jobs)) ||
        ((output || open_viewer) && !profiling) ||
        (building && forwarded < argc)) {
        nob_log(NOB_ERROR, "Invalid compiler, mode or options for %s. Use --help", action);
        return 1;
    }
    if (profiling) {
        for (int i = forwarded; i < argc; ++i) {
            if (strcmp(argv[i], "--validate-only") == 0) {
                nob_log(NOB_ERROR, "nob profile validates automatically; omit --validate-only");
                return 1;
            }
        }
        if (strcmp(mode, "profile") != 0) {
            nob_log(NOB_ERROR, "profile uses --mode profile (optimized with symbols)");
            return 1;
        }
#ifdef _WIN32
        if (strcmp(cc, "clang-cl") != 0 && strcmp(cc, "msvc") != 0) {
            nob_log(NOB_ERROR, "Windows Samply captures need PDB symbols; use --cc clang-cl or --cc msvc");
            return 1;
        }
#endif
        if (!output) output = capture_path();
        if (!output) return 1;
        size_t length = strlen(output);
        size_t base_length = length >= 3 && strcmp(output + length - 3, ".gz") == 0 ? length - 3 : length;
        const char *symbols = nob_temp_sprintf("%.*s.syms.json", (int)base_length, output);
        if (nob_file_exists(output) != 0 || nob_file_exists(symbols) != 0 ||
            nob_file_exists(nob_temp_sprintf("%s.command.txt", output)) != 0 ||
            nob_file_exists(nob_temp_sprintf("%s.stdout.log", output)) != 0 ||
            nob_file_exists(nob_temp_sprintf("%s.stderr.log", output)) != 0) {
            nob_log(NOB_ERROR, "Capture path or sidecars already exist; choose a fresh --output path");
            return 1;
        }
    }

    const char *directory = nob_temp_sprintf("build/%s/%s", cc, mode);
    log_directory = nob_temp_sprintf("%s/logs", directory);
    if (!nob_mkdir_if_not_exists("build") ||
        !nob_mkdir_if_not_exists(nob_temp_sprintf("build/%s", cc)) ||
        !nob_mkdir_if_not_exists(directory) ||
        !nob_mkdir_if_not_exists(log_directory)) return 1;
    fprintf(stderr, "[build] %s / %s\n", cc, mode);
    const char *compiler = strcmp(cc, "msvc") == 0 ? "cl" : cc;
    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "cmake", "-S", ".", "-B", directory, "-G", "Ninja",
                   nob_temp_sprintf("-DCMAKE_C_COMPILER=%s", compiler),
                   nob_temp_sprintf("-DCMAKE_BUILD_TYPE=%s", configuration),
                   "-DBUILD_TESTING=ON");
    if (!run_setup(&cmd, "configure")) return 1;
    nob_cmd_append(&cmd, "cmake", "--build", directory, "--parallel");
    if (jobs) nob_cmd_append(&cmd, jobs);
    if (!run_setup(&cmd, "build")) return 1;
    if (building) {
        fprintf(stderr, "Build succeeded. Logs: %s/\n", log_directory);
        return 0;
    }
    fprintf(stderr, "\n[%s]\n", testing ? "tests" : profiling ? "profile" : "benchmark");

    if (profiling) {
        /* Validate the exact optimized build before starting the profiler. */
        nob_cmd_append(&cmd, "ctest", "--test-dir", directory,
                       "--output-on-failure", "--no-tests=error");
        if (!run_setup(&cmd, "tests")) return 1;
        fprintf(stderr, "[check] Correctness tests passed.\n");
    }

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
        if (profiling)
            nob_cmd_append(&cmd, "--profile", "--impl", "naive", "--repeats", "200");
    }
    for (int i = forwarded; i < argc; ++i) nob_cmd_append(&cmd, argv[i]);
    /* Tool help/listing should not start a capture. */
    bool information_only = false;
    for (int i = forwarded; i < argc; ++i)
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "--list") == 0)
            information_only = true;
    bool ok = profiling && !information_only ? capture(&cmd, output, open_viewer) :
                                              nob_cmd_run(&cmd);
    nob_cmd_free(cmd);
    return ok ? 0 : 1;
}
