= knn image classifier
== by Lucien Rohwein Aibel and Griffin Gooch-Breault

A demonstration of different parallelism approaches in Rust, through implimenting a k-nearest neighbor image classifier. Takes in a dataset of 28x28 greyscale images of digits (such as the MNIST Handwritten Digits dataset), running the k-nearest neighbor algorithm on each image in /test using all the images in /train as the neighbors to compare with (these directories need to have ten subdirectories, one for each digit). Reports accuracy and time, and supports sequential, threaded, or rayon-based approaches.

== Prerequisites

Built using rust 1.93.1
Other dependencies in cargo.toml
Reccomend at least 60 MB of memory, currently untested on systems with less than eight cores but should work

== Set-up

1. Download and unpack the code
2. Download a dataset
We used MNIST Handwritten Digits, retrieved from https://github.com/rasbt/mnist-pngs
3. Ensure dataset has a /train and a /test directory, with each having one subdirectory for each digit 0-9.
4. Run code with cargo run --release help to get syntax help
