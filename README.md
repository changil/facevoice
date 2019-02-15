# On Learning Associations of Faces and Voices

This repository contains a single-file, reference implementation of the following publication:

> **On Learning Associations of Faces and Voices**  
> Changil Kim, Hijung Valentina Shin, Tae-Hyun Oh, Alexandre Kaspar, Mohamed Elgharib, Wojciech Matusik  
> *ACCV 2018*  
> [Paper](https://people.csail.mit.edu/changil/assets/face-voice-accv-2018.pdf) | [ArXiv](https://arxiv.org/abs/1805.05553) | [Project Website](http://facevoice.csail.mit.edu/)

Please cite the above paper if you use this software. See the project website for more information about the paper.

## Requirements

The software runs with Python 2 or 3, and TensorFlow r1.4 or later. Additionally, it requires NumPy, SciPy, and scikit-image packages.

## Pre-trained models

Two pre-trained models are provided as TensorFlow checkpoints.

* [Model trained with voice as the reference modality (v2f)](http://facevoice.csail.mit.edu/facevoice-checkpoint-v2f.zip)
* [Model trained with face as the reference modality (f2v)](http://facevoice.csail.mit.edu/facevoice-checkpoint-f2v.zip)

## Usage

Download pre-trained models and unzip them. Prepare input facial images and voice files: facial images must be JPEG or PNG color images, and audio files must be WAV audio files sampled at 22,050 hz.

Depending on the reference modality, run one of the following two commands. Make sure you specify the correct checkpoint matching the reference modality.

* Given a voice, find the matching face from two candidates (v2f):
  ```bash
  facevoice.py v2f -c CHECKPOINTDIR --voice VOICEFILE --face0 FACEFILE --face1 FACEFILE
  ```

* Given a face, find the matching voice from two candidates (f2v):
  ```bash
  facevoice.py f2v -c CHECKPOINTDIR --face FACEFILE --voice0 VOICEFILE --voice1 VOICEFILE
  ```

