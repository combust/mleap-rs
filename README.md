# MLeap on Rust

MLeap is a runtime for executing machine learning pipelines created
using Spark, Scikit-learn or Tensorflow. It includes a JSON/Protobuf3
serialization format as well as an execution engine for scoring results
in realtime. It is focused on performance and portability.

The MLeap on Rust project aims to expand the capabilities of the
original [MLeap](https://github.com/combust/mleap) by taking it off of the JVM.

## Requirements

1. Install [rustup](https://www.rustup.rs/)
2. Install the stable rust: `rustup install stable`
3. `git submodule init`
4. `git submodule update`

## Building

After installing the stable version of rust with rustup, it is easy to
build MLeap:

```
cargo build
```

## Running Tests

The tests require a pretrained Airbnb model that can be downloaded here:
[airbnb.model.json.zip](https://s3-us-west-2.amazonaws.com/mleap-demo/airbnb.model.json.zip).

Once you have it downloaded:

1. `mkdir /tmp/model`
2. Copy the model file into `/tmp/model`
3. Unzip the model file, the tests expect it to be here
4. Make sure that `/tmp/model/bundle.json` exists. This is a good check
   to make sure the model is good and was unzipped properly to the
`/tmp/model` folder.
5. `cargo test`

## Examples

The tests are good examples for how to work with MLeap on Rust.
Take look at [the tests](https://github.com/combust/mleap-rs/blob/master/src/bundle/mod.rs).

The tests will show how to build a LeapFrame from scratch, load a
transformer from an MLeap bundle, transform the LeapFrame and then
extract a result.

1. `test_airbnb` shows the Rust interface
2. `test_airbnb_c` show the C native interface to the MLeap library

## C Native Interface

The C native interface is a collection of C-compatible functions exposed
by MLeap. It is defined in one file here: [c/mod.rs](https://github.com/combust/mleap-rs/blob/master/src/c/mod.rs).

LeapFrames are passed as `*mut frame::LeapFrame`, these can just be
`void *` in C code.

Transformers are passed as `*mut Box<tform::DefaultNode>`, again, in C
code these can just be `void *` when declaring the method signature.

### Resource/Memory Management

There are two methods that allocate resources:
1. `mleap_frame_with_size` allocates a new LeapFrame
2. `mleap_transformer_load` allocates a new Transformer

In order to free these resources when you are done, make sure to use
these two corresponding methods:

1. `mleap_frame_free` to free a LeapFrame, this will also deallocate any
   data stored in the columns
2. `mleap_transformer_free` to free a Transformer

Forgetting to free resources will cause a memory leak.
