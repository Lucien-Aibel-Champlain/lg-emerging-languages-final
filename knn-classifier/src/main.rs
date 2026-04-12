use crate::knn_model::{KNNModel};
use crate::sequential_model::{SequentialModel};
use crate::rayon_model::{RayonModel};
use std::env;

pub mod mnist_image;
pub mod knn_model;
pub mod sequential_model;
pub mod rayon_model;

const DEFAULT_THREADS: usize = 1; //default to sequential

fn help(command_for_this: &str) {
    println!("Syntax: {} [directory target] [optional: number of threads]",command_for_this);
    println!("");
    println!("Directory targeted should have two subdirectories, /test and /train. Each should have ten subdirectories, one for each digit as a numeral (e.g. /0 or /1)");
    println!("");
    println!("If number of threads is not specified, defaults to sequential (single-threaded).");
    println!("Thread count includes the main thread; so count 4 spawns a maximum of 3 threads alongside the main one that awaits their return.");
}

fn collect_arguments() -> Result<(String, usize), String> {
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

    Ok((directory_target.clone(), thread_count))
}

fn main() {
    let (directory_target, thread_count) = if let Ok(result) = collect_arguments() { result } else { return };

    if thread_count <= 1 {
        println!("Loading sequential model ...");
        SequentialModel::load_and_test(&directory_target);
    } else {
        println!("Loading rayon-parallel model ...");
        let threads_beyond_main = thread_count - 1;
        rayon::ThreadPoolBuilder::new().num_threads(threads_beyond_main).build_global().expect("Error setting rayon parameters"); //set maximum threads across all rayon pools
        RayonModel::load_and_test(&directory_target);
    }
}
