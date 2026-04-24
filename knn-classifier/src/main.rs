use crate::knn_model::{KNNModel};
use crate::sequential_model::{SequentialModel};
//use crate::rayon_model::{RayonModel};
use crate::stdthread_model::{StdThreadModel};
use std::env;

pub mod mnist_image;
pub mod knn_model;
pub mod sequential_model;
pub mod rayon_model;
pub mod stdthread_model;

const DEFAULT_THREADS: usize = 1; //default to sequential
const DEFAULT_VERBOSITY: bool = false;

fn help(command_for_this: &str) {
    println!("Syntax: {} [directory target] [optional: number of threads] [optional: verbose y/n]",command_for_this);
    println!("");
    println!("Directory targeted should have two subdirectories, /test and /train. Each should have ten subdirectories, one for each digit as a numeral (e.g. /0 or /1)");
    println!("");
    println!("If number of threads is not specified, defaults to sequential (single-threaded).");
    println!("Thread count includes the main thread; so count 4 spawns a maximum of 3 threads alongside the main one that awaits their return.");
    println!("");
    println!("Verbosity, if specified should be either 'y' for yes or 'n' for no.");
    println!("When enabled, every result will be printed as soon as it is processed.");
    println!("Default: {}.", DEFAULT_VERBOSITY);
}

fn collect_arguments() -> Result<(String, usize, bool), String> {
    let args: Vec<String> = env::args().collect();

    let command_for_this = args.get(0).expect("Expected to find command used to run program, found nothing.");

    let directory_target = match args.get(1) {
        Some(arg) => arg,
        None => {
            println!("Invalid syntax! Type '{} help' for help.", command_for_this);
            return Err(String::new());
        },
    };
    if directory_target == "help" {
        help(command_for_this);
        return Err(String::new());
    }

    let thread_count = match args.get(2) {
        Some(arg) => match arg.parse::<usize>() {
            Ok(arg) => arg,
            Err(_) => {
                println!("{} is not a valid number of threads.", arg);
                return Err(String::new());
            }
        },
        None => DEFAULT_THREADS,
    };

    let verbose = match args.get(3) {
        Some(arg) => match arg.trim().to_lowercase().as_str() {
            "y" | "t" => true,
            "n" | "f" => false,
            other => {
                println!("Invalid setting {} for verbosity. Must be either 'y' or 'n'.", other);
                return Err(String::new());
            },
        },
        None => DEFAULT_VERBOSITY
    };

    Ok((directory_target.clone(), thread_count, verbose))
}

fn main() {
    let (directory_target, thread_count, verbose) = if let Ok(result) = collect_arguments() { result } else { return };

    if thread_count <= 1 {
        println!("Loading sequential model ...");
        SequentialModel::load_and_test(&directory_target, verbose);
    } else {
        println!("Loading stdthread model ...");
        let threads_beyond_main = thread_count - 1;

        StdThreadModel::load_and_test(&directory_target, verbose);
    }
}
