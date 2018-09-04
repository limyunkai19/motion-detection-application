# Motion Detection Application
An object motion detection application with background subtraction technique. Implemented with C++ and OpenCV. A project submitted to my university subject, UCCC2513 Mini Project.

## Features
- Compute the background after a training phase
- Support dynamic background (background with movement, see results below)
- Compute the foreground mask
- Compute bounding box of objects
- Compute path of moving objects
- Compute velocity of moving objects

## Results
**Video Sample Input 1 - Static Background**

| Input Frame | Result |
|:-----------:|:------:|
| **Background** | ![bg1](result-img/bg1.jpg) |
| ![in1](result-img/in1.jpg) | ![mask1](result-img/mask1.jpg) |
| ![in1](result-img/in1.jpg) | ![res1](result-img/res1.jpg) |
| ![in2](result-img/in2.jpg) | ![res2](result-img/res2.jpg) |

<br>
<hr>
<br>

**Video Sample Input 2 - Dynamic Background (Moving Fountain Water)**

| Input Frame | Result |
|:-----------:|:------:|
| **Background** | ![bg2](result-img/bg2.jpg) |
| ![in3](result-img/in3.jpg) | ![mask3](result-img/mask3.jpg) |
| ![in3](result-img/in3.jpg) | ![res3](result-img/res3.jpg) |
| ![in4](result-img/in4.jpg) | ![res4](result-img/res4.jpg) |

## Improvement
- [ ] usage of LAB color space
- [ ] adaptive background
