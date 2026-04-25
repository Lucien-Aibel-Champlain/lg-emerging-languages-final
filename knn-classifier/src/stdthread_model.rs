use std::fs;
use std::thread;
use std::mem::{swap};
use std::sync::mpsc::{channel, Receiver, Sender};
use crate::mnist_image::{MNISTImage, ClassedImage};
use crate::knn_model::{KNNModel};

pub struct StdThreadModel {
    dataset: Vec<ClassedImage>,
    threads: Vec<(Option<thread::JoinHandle<()>>, Sender<Option<MNISTImage>>)>,
    return_channel: Receiver<(f64, u8)>,
    return_channel_sender: Sender<(f64, u8)>,
}

impl StdThreadModel {

    //Spawn threads with unique slice of the training data
    fn initialize(&mut self, num_threads: usize) {
        for thread_id in 0..num_threads {

            //for verbose?
            //println!("\n Spawning thread {} \n responsable for training data slice {}-{}", thread_id, (thread_id*(self.len()/self.number_of_threads)), (thread_id+1)*(self.len()/self.number_of_threads));

            // get slice of training data for thread at thread_id
            let dataset_slice: Vec<ClassedImage> = if thread_id == num_threads - 1 {
                self.dataset[thread_id*(self.len()/num_threads)..self.len()].to_vec()
            } else {
                self.dataset[thread_id*(self.len()/num_threads)..(thread_id+1)*(self.len()/num_threads)].to_vec()
            };
            
            // make channel for input
            let (working_image_tx, working_image_rx) = channel::<Option<MNISTImage>>();
            // clone sender of output channel
            let local_return_channel_sender = self.return_channel_sender.clone();

            let handle = thread::spawn( move || {
                //for verbose?
                //println!("Spawned {}, covering slice size: {}", thread_id, dataset_slice.len());
                loop{

                    //for verbose?
                    //println!("thread {}, awaiting", thread_id);
                    let test_image: MNISTImage;

                    //Block thread until recv, if we get None the thread will finish its process (this isnt the best way to do this)
                    match working_image_rx.recv() {
                        Ok(message) => {
                            match message {
                                Some(image) => test_image = image,
                                None => {
                                    break
                                },
                            }
                        },
                        Err(_) => break,
                    }

                    //iterate through training data, calculate distance and send it off via the return channel
                    for training_image in &dataset_slice {
                        let _ = local_return_channel_sender.send((test_image.calculate_distance(&training_image.image), training_image.class));
                    }
                    
                    // send impossible result as marker for end of thread process (again sub-par method to signal that the thread is done, but I've neglected too much other work just to make this compile, I'm not spending more time here. Even if it would be an easy fix tho)
                    let _ = local_return_channel_sender.send((f64::INFINITY, 255 )); 
                }

                //for verbose?
                //println!("Thread {} dropping", thread_id);
            });

            //Once the thread is spawned we add relevent data (its joinhandle and input channel sender) to the threads Vec
            self.threads.push((Some(handle), working_image_tx));
        }
    }

    pub fn read_and_initialize(directory: &str, num_threads: usize) -> Result<StdThreadModel, String> {
        match StdThreadModel::from_directory(directory) {
            Ok(mut new_model) => {
                new_model.initialize(num_threads);
                Ok(new_model)
            }
            Err(msg) => Err(msg),
        }
    }
}

//This needs to exist to stop the threads from becoming detached.
//All it does is send None to the input channel to get the threads to finish, then we join them
impl Drop for StdThreadModel{
    fn drop (&mut self){
        for thread in &mut self.threads{
            let _ = thread.1.send(None); // not sure that this is best practice but the channels would have already been contacted
            let mut handle: Option<thread::JoinHandle<()>> = None;

            swap(&mut handle, &mut thread.0);
            let _ = match handle {
                Some(join_handle) => {
                    join_handle.join()
                },
                None => continue,
            };
        }
    }
}

impl KNNModel for StdThreadModel {
    #[allow(refining_impl_trait)]
    fn new() -> StdThreadModel {
        //This is the only value created in the new method, we need to wait for the training data to load before we can spawn the threads.
        let (return_channel_tx, return_channel_rx) = channel::<(f64, u8)>(); 
        return StdThreadModel {
            dataset: Vec::new(),
            //would be ideal to work this into the flow of the larger struct
            threads: Vec::new(),
            return_channel: return_channel_rx,
            return_channel_sender: return_channel_tx,
        };
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    //Should not be used on its own; initialize must be called after. Use read_and_initialize to call both.
    #[allow(refining_impl_trait)]
    fn from_directory(directory: &str) -> Result<StdThreadModel, String> {
        let training_directory = directory.to_owned() + "/train/";
        
        let mut new_model = StdThreadModel::new();

        for digit in 0..=9 {
            //Create a new string (cloned from the general training string) to point to this digit's subfolder
            let current_directory = training_directory.clone() + &digit.to_string();

            //Get a reader to list all entries inside current_directory
            let reader = match fs::read_dir(&current_directory) {
                Ok(unpacked_reader) => unpacked_reader,
                Err(_) => return Err(format!("Cannot open {} for reading", current_directory)),
            };

            let digitset: Vec<Result<MNISTImage, String>> = reader.map(|entry| Self::entry_to_img(&entry, &current_directory)).collect();
            for img_result in digitset {
                match img_result {
                    Ok(img) => new_model.dataset.push(ClassedImage {
                        image: img,
                        class: digit,
                    }),
                    Err(msg) => return Err(msg),
                };
            }
        }

        Ok(new_model)
    }

    fn classify(&self, image: &MNISTImage, k: u32) -> Result<u8, String> {
        // send out message to all threads via their individual input channels
        for (_, input_channel) in &self.threads {
            match input_channel.send(Some(image.clone())){
                Ok(_) => (),
                Err(err) => panic!("The image could not be sent to threads {}", err),
            };
        }

        // distances vec is filled with impossible data 
        let mut distances: Vec<(f64, u8)> = vec![(f64::INFINITY, 255); usize::try_from(k).unwrap()];

        // this is used to create an exit condition after an image has been tested against all of the data
        let mut threads_complete: usize = 0;
        
        loop {

            // close loop after all threads are done with work
            if threads_complete >= self.threads.len() {
                break;
            }

            let message = self.return_channel.recv();

            match message {
                Ok((distance, class)) => {
                    if distance == f64::INFINITY {
                        // if a thread returns impossible data we know its done and skip it
                        threads_complete += 1;
                        continue; 
                    }
                    // linear sort capped at k (should be revisited)
                    for i in 0..k {
                        if distance < distances[usize::try_from(i).unwrap()].0 {
                            distances.insert(usize::try_from(i).unwrap(), (distance, class));
                            break;
                        }
                    }
                },
                Err(err) => println!("Failed to read response from thread, error: {}", err),
            }



        }
        
        distances.truncate(usize::try_from(k).unwrap());
        return Ok(Self::take_votes(distances))
    }
}