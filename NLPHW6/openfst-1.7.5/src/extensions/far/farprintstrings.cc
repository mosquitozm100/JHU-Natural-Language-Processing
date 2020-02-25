#include <fst/flags.h>

DEFINE_string(filename_prefix, "", "Prefix to append to filenames");
DEFINE_string(filename_suffix, "", "Suffix to append to filenames");
DEFINE_int32(generate_filenames, 0,
             "Generate N digit numeric filenames (def: use keys)");
DEFINE_string(begin_key, "",
              "First key to extract (def: first key in archive)");
DEFINE_string(end_key, "", "Last key to extract (def: last key in archive)");
// PrintStringsMain specific flag definitions.
DEFINE_bool(print_key, false, "Prefix each std::string by its key");
DEFINE_bool(print_weight, false, "Suffix each std::string by its weight");
DEFINE_string(entry_type, "line",
              "Entry type: one of : "
              "\"file\" (one FST per file), \"line\" (one FST per line)");
DEFINE_string(token_type, "symbol",
              "Token type: one of : "
              "\"symbol\", \"byte\", \"utf8\"");
DEFINE_string(symbols, "", "Label symbol table");
DEFINE_bool(initial_symbols, true,
            "Uses symbol table from the first Fst in archive for all entries.");

int farprintstrings_main(int argc, char **argv);

int main(int argc, char **argv) {
  return farprintstrings_main(argc, argv);
}
