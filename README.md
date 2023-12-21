# lf-fp-tools
Tools specifically developed for evaluating some features of Lingua Franca running on FlexPRET.

# Getting started

The tools are stand-alone, meaning the Lingua Franca and FlexPRET repositories are needed as well. An option is to use https://github.com/icyphy/lf-flexpret. 

## gen_PipelineN.py

Generates an Lf program that uses N pipelines. Takes exactly one argument, which is the number of stages `N`. Other optional parameters are:
* the #define MAX_ITERATIONS can be changed to a smaller number for simple testing (line 26).
* the LF program is written to a hard-coded file path; feel free to change this (line 148).

For earlier results, N = 1000 has been used. This data can take a while to gather, however, beacuse the emulated printing is slow on FlexPRET.

## LF program printing format

Each of the benchmark programs `Ìnterrupts.lf`, `ScatterGather.lf`, and `Pipeline<N>.lf` print in the same format, e.g.,

`[1]: Iteration 8: Base: 140192780, {0}: 141023260, {1}: 141114790, {2}: 141135940, {3}: 141153710, {4}: 141187080, {5}: 141171530, {6}: 141080050,`

The `analyze.py` script expects some variation of this format (including the trailing comma). Do not change the source code for this printing format and everything should be ok.

## analyze.py

Takes in one or several .txt files on the mentioned format and analyzes the data. Expects files from paths on the format:

`results/<model>/w<number of workers>it<number of iterations>.txt`

e.g.,

`results/pipeline/w5it1000.txt`.

This path and/or format can be changed on line 303.
And yes, the code here is a mess...

# Usage

1. Build FlexPRET with the desired number of hardware threads. In the lf-flexpret repository, the makefile derives the number of worker threads from the build number of hardware threads (one less than the number of hardware threads).
2. Compile the LF programs. Make sure the number of iterations is as desired (#define MAX_INTERRUPTS, MAX_ITERATIONS in preamble).
3. Run the LF programs. Redirect their standard outputs to files, like so: `fp-emu +ispm=ScatterGather.mem > results/w5it1000.txt` (if there are five workers and 1000 iterations). Note that this procedure takes a while. It is a good idea to try with e.g., 10 iterations first to check that it works as expected.
4. For the interrupt program, there needs to be a client that triggers interrupts as well. Use `--client` argument for `fp-emu` and run the `flexpret/emulator/clients/build/interrupter -a -n <number of interrupts> -d <delay_ms>` to automatically send a number of interrupts. The client's time is not synchronized with FlexPRET's time - i.e., 1 ms on FlexPRET is not the same as 1 ms for the client, because simulating FlexPRET for 1 ms takes longer than 1 ms in the real world.
