use crate::knn_model::{KNNModel};
use crate::sequential_model::{SequentialModel};
use crate::rayon_model::{RayonModel};
use crate::stdthread_model::{StdThreadModel};
use std::env;
use std::time::Instant;

pub mod mnist_image;
pub mod knn_model;
pub mod sequential_model;
pub mod rayon_model;
pub mod stdthread_model;

#[derive(PartialEq)]
enum ModelType {
    Sequential,
    StdThread,
    Rayon,
}
const DEFAULT_MODEL_TYPE: ModelType = ModelType::Sequential;

const DEFAULT_THREADS: usize = 1; //default to sequential
const DEFAULT_VERBOSITY: bool = false;

fn help(command_for_this: &str) {
    println!("Syntax: {} [directory target] [k] [optional: model to use] [optional: number of threads] [optional: verbose y/n]",command_for_this);
    println!("");
    println!("Directory targeted should have two subdirectories, /test and /train. Each should have ten subdirectories, one for each digit as a numeral (e.g. /0 or /1)");
    println!("");
    println!("k is the number of nearest neighbors that get a vote. Only parameter that changes accuracy.");
    println!("");
    println!("Model to use can be one of 's' for sequential, 't' for threaded, or 'r' for rayon.");
    println!("Defaults to {} if not specified.", match DEFAULT_MODEL_TYPE {
        ModelType::Sequential => "sequential",
        ModelType::StdThread => "stdthread",
        ModelType::Rayon => "rayon",
    });
    println!("");
    println!("If number of threads is not specified, defaults to {}.", DEFAULT_THREADS);
    println!("Thread count includes the main thread; so count 4 spawns a maximum of 3 threads alongside the main one that awaits their return.");
    println!("");
    println!("Verbosity, if specified should be either 'y' for yes or 'n' for no.");
    println!("When enabled, every result will be printed as soon as it is processed.");
    println!("Default: {}.", DEFAULT_VERBOSITY);
}

fn collect_arguments() -> Result<(String, u32, usize, ModelType, bool), String> {
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

    let k = match args.get(2) {
        Some(arg) => match arg.parse::<u32>() {
            Ok(arg) => arg,
            Err(_) => {
                println!("{} is not a valid number for k.", arg);
                return Err(String::new());
            }
        },
        None => {
            println!("Must specify a value for k.");
            return Err(String::new());
        },
    };

    let model_type = match args.get(3) {
        Some(arg) => match arg.trim().to_lowercase().as_str() {
            "s" => ModelType::Sequential,
            "t" => ModelType::StdThread,
            "r" => ModelType::Rayon,
            other => {
                println!("Invalid setting {} for model type. Must be 's', 't', or 'r'.", other);
                return Err(String::new());
            },
        },
        None => DEFAULT_MODEL_TYPE
    };

    let thread_count = match args.get(4) {
        Some(arg) => match arg.parse::<usize>() {
            Ok(arg) => arg,
            Err(_) => {
                println!("{} is not a valid number of threads.", arg);
                return Err(String::new());
            }
        },
        None => DEFAULT_THREADS,
    };

    if model_type == ModelType::Sequential && thread_count != 1 {
        println!("Model type sequential must have number of threads 1.");
        return Err(String::new());
    }

    let verbose = match args.get(5) {
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

    Ok((directory_target.clone(), k, thread_count, model_type, verbose))
}

fn main() {
    let (directory_target, k, thread_count, model_type, verbose) = if let Ok(result) = collect_arguments() { result } else { return };

    let t0 = Instant::now();
    let model: Box<dyn KNNModel> = 
        if model_type == ModelType::StdThread && thread_count > 1 {
            println!("Loading stdthread model ...");
            let threads_beyond_main = thread_count - 1;
            match StdThreadModel::read_and_initialize(&directory_target, threads_beyond_main) {
                Ok(model) => Box::new(model),
                Err(msg) => {
                    println!("Error loading data: {}",msg);
                    return;
                }
            }
        } else if model_type == ModelType::Rayon && thread_count > 1 {
            println!("Loading rayon model ...");
            let threads_beyond_main = thread_count - 1;
            rayon::ThreadPoolBuilder::new().num_threads(threads_beyond_main).build_global().expect("Error setting rayon parameters"); //set maximum threads across all rayon pools
            match RayonModel::from_directory(&directory_target) {
                Ok(model) => Box::new(model),
                Err(msg) => {
                    println!("Error loading data: {}",msg);
                    return;
                }
            }
        }
        else {
            println!("Loading sequential model ...");
            match SequentialModel::from_directory(&directory_target) {
                Ok(model) => Box::new(model),
                Err(msg) => {
                    println!("Error loading data: {}",msg);
                    return;
                }
            }
        };
    let t0 = t0.elapsed().as_secs_f64();
    println!("Data imported successfully! Length: {}", model.len());
    println!("Loaded in {}s", t0);

    let test_t0 = Instant::now();
    match model.test(k, &directory_target, verbose) {
            Ok(score) => println!("\nTest complete.\nAccuracy: {}%", score * 100.0),
            Err(string) => println!("\nError while testing: {}",string),
    }
    let test_t0 = test_t0.elapsed().as_secs_f64();
    println!("Test complete after {} seconds.", test_t0);
}
