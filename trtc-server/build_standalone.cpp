// The standalone builder: exactly the server's `build` subcommand, as its own
// binary — for one-shot local builds (nix run .#build-10.13) and anywhere the
// broker isn't wanted.

int build_main(int argc, char **argv);

int main(int argc, char **argv) { return build_main(argc, argv); }
