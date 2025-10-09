CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Griffin Evans
* Tested on: Windows 11 Education, i9-12900F @ 2.40GHz 64.0GB, NVIDIA GeForce RTX 3090 (Levine 057 SIGLAB-02 (though marked 3 on the label))

![](img/chessDuck.2025-10-05_23-06-26z.1639samp.png)
![](img/bespokeScene.2025-10-08_00-54-20z.5000samp.png)
![](img/bespokeScene.2025-10-07_23-56-20z.3718samp.png)
![](img/chessDuck.2025-10-05_23-46-18z.397samp.png)

## Features and Analysis

### Materials—Diffusion, Reflection, and Refraction with Fresnel effects

<img width="500" height="500" alt="cornell 2025-10-08_20-51-37z 982samp" src="https://github.com/user-attachments/assets/1c02346e-25ea-4b85-8517-82eeaf20acd3" />
<img width="500" height="500" alt="cornell 2025-10-08_19-08-18z 2330samp" src="https://github.com/user-attachments/assets/af8c6cb2-f0b5-4997-ab92-7af1e5328238" />
<img width="500" height="500" alt="cornell 2025-10-08_18-40-25z 1250samp" src="https://github.com/user-attachments/assets/78525808-204d-4cf3-be37-4af87a2ded3c" />
<img width="500" height="500" alt="cornell 2025-10-08_18-41-13z 5000samp" src="https://github.com/user-attachments/assets/ea0fd7b8-ae5e-406e-9c4e-c022bc268dcb" />

<img width="750" height="1001" alt="Screen Shot 2025-10-08 at 7 25 21 PM" src="https://github.com/user-attachments/assets/fc4e2d4b-f3dc-4898-a7af-18897c8544b4" />


### Motion blur

<img width="500" height="500" alt="cornell 2025-10-08_20-51-37z 982samp" src="https://github.com/user-attachments/assets/867da07a-4ab9-490a-9d1c-915c11c0d1ac" />
<img width="500" height="500" alt="cornell 2025-10-08_18-43-29z 5000samp" src="https://github.com/user-attachments/assets/d83d7940-060a-4d25-b211-db2ea327ab0c" />

<img width="500" height="500" alt="duckMotion 2025-10-08_18-48-03z 5000samp" src="https://github.com/user-attachments/assets/10146886-6934-4901-aaf6-c6a54d30e91c" />
<img width="500" height="500" alt="duckMotion 2025-10-08_18-45-46z 5000samp" src="https://github.com/user-attachments/assets/f90dba97-95bc-48ce-8130-d9cdd34d3095" />

<img width="1585" height="979" alt="Average kernel run times per frame for scenes with and without motion blur keyframes (lower is better)" src="https://github.com/user-attachments/assets/60afa2b3-9828-4d55-a412-934ac1d76351" />



### Physically-based depth-of-field
<img width="500" height="500" alt="cornell 2025-10-08_18-33-51z 323samp" src="https://github.com/user-attachments/assets/7535d81f-0b6e-4157-b313-0116550d3f34" />
<img width="500" height="500" alt="cornell 2025-10-08_18-33-51z 353samp" src="https://github.com/user-attachments/assets/7806cb44-75f4-4e8b-8031-42cd33bc27cd" />
<img width="500" height="500" alt="cornell 2025-10-08_18-36-11z 127samp" src="https://github.com/user-attachments/assets/048ff775-a772-41ac-816e-5521166ec822" />
<img width="500" height="500" alt="cornell 2025-10-08_18-36-11z 95samp" src="https://github.com/user-attachments/assets/783ed42c-b461-4103-84f0-f4118b11c3bc" />
<img width="750" height="1304" alt="Average kernel run times per frame for box and sphere scene with and without depth of field (lower is better)" src="https://github.com/user-attachments/assets/c3d5f171-2572-4a5c-ae4d-a820bd8035e2" />


### Arbitrary mesh rendering via glTF

### BVH spatial data structure
<img width="750" height="750" alt="cornell 2025-10-08_20-30-35z 797samp" src="https://github.com/user-attachments/assets/434fddf2-343d-4dd6-8f33-4f10ac36b38f" />
<img width="750" height="1268" alt="Average kernel run times per frame for glTF duck model (lower is better) (1)" src="https://github.com/user-attachments/assets/0677006e-5ab0-4859-ad2a-5f26b92125d4" />

### Environment mapping
<img width="330" height="330" alt="bespokeScene 2025-10-08_20-26-40z 106samp" src="https://github.com/user-attachments/assets/d44b9a6f-80aa-4ed6-bb3f-345a274a0e3e" />
<img width="330" height="330" alt="bespokeScene 2025-10-08_20-28-06z 36samp" src="https://github.com/user-attachments/assets/a25cc897-f4bc-4d83-a569-c3585ce12b42" />
<img width="330" height="330" alt="bespokeScene 2025-10-08_20-28-53z 58samp" src="https://github.com/user-attachments/assets/26278f99-dcce-4722-a49f-d8d2ff144135" />

<img width="330" height="330" alt="cornell 2025-10-08_20-33-57z 389samp" src="https://github.com/user-attachments/assets/5ac0aa05-ba72-4a9a-8786-fb2fe9014584" />
<img width="330" height="330" alt="cornell 2025-10-08_20-35-04z 2307samp" src="https://github.com/user-attachments/assets/e1616818-617f-44e4-87bd-d4df7446478c" />
<img width="330" height="330" alt="cornell 2025-10-08_20-35-47z 3842samp" src="https://github.com/user-attachments/assets/447b938c-dad4-4c85-aad7-4ae7ab6aeb50" />

<img width="1000" height="979" alt="Average run time per frame of shadeMaterial kernel for varying background environments (ms; lower is better)" src="https://github.com/user-attachments/assets/c4c9f7c8-2d35-4357-8e13-9418388cac20" />



### Stream Compaction and Material Sorting
<img width="330" height="330" alt="cornell 2025-10-08_20-35-47z 3842samp" src="https://github.com/user-attachments/assets/afa056f0-5f21-4dca-8b3a-f0e39b25f894" />
<img width="330" height="330" alt="bespokeScene 2025-10-07_23-56-20z 3718samp" src="https://github.com/user-attachments/assets/7b8feecf-f994-4088-8935-c2a523901a39" />
<img width="330" height="330" alt="bespokeScene 2025-10-08_20-22-42z 38samp" src="https://github.com/user-attachments/assets/a39a4af4-91bf-4211-b23e-40d3c3bdc39e" />
<img width="750" height="1154" alt="Average kernel run times per frame (summed over all depths) for Cornell box scene with and without stream compaction and material sorting (lower is better)" src="https://github.com/user-attachments/assets/ab13f518-20fa-47e2-bcf7-1a03f99e225c" />
<img width="750" height="931" alt="Average kernel run times for Cornell box scene at depth 5 and 6 with and without compaction and sorting" src="https://github.com/user-attachments/assets/b17f3fbe-0cac-49ed-b7ab-19944fdefa08" />

### Russian roulette path termination




## Third Party Sources
I made use of the [tinygltf library](https://github.com/syoyo/tinygltf/) as suggested in order to load in model data from glTF files. I feel I should note I did look at Ruipeng Wang's code from last year in regards to using the aforementioned tinygltf library when I was talking with him in office hours about trying to fix a bug I was having with loading the vertex indices.

I also referenced [the article "How to build a BVH" by Jacco Bikker](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/) in order to understand how to implement the BVH construction and traversal algorithms.

Reused timer code from Stream Compaction project for performance testing, due to having issues running Nsight on the machine I was using.

Environment map textures used are [Dam Wall, by Dimitrios Savva and Jarod Guest](https://polyhaven.com/a/dam_wall), and [Industrial Sunset (Pure Sky), by Sergej Majboroda and Jarod Guest](https://polyhaven.com/a/industrial_sunset_puresky), both from Poly Haven.

All glTF files used were either created by me using Maya (those in the `scenes/glTF` folder) or obtained from [the Khronos Group's sample asset repository](https://github.com/KhronosGroup/glTF-Sample-Assets) (those in the `scenes/glTF-Samples` folder):

"A Beautiful Game" Credit:
© 2020, ASWF. CC BY 4.0 International
- MaterialX Project for Original model
© 2022, Ed Mackey. CC BY 4.0 International
- Ed Mackey for Conversion to glTF

"Avocado" Credit:
© 2017, Public. CC0 1.0 Universal
- Microsoft for Everything

"Box" Credit:
© 2017, Cesium. CC BY 4.0 International
- Cesium for Everything

"Duck" Credit:
© 2006, Sony. SCEA Shared Source License, Version 1.0
- Sony for Everything

"Fox" Credit:
© 2014, Public. CC0 1.0 Universal
- PixelMannen for Model
© 2014, tomkranis. CC BY 4.0 International
- tomkranis for Rigging & Animation
© 2017, @AsoboStudio and @scurest. CC BY 4.0 International
- @AsoboStudio and @scurest for Conversion to glTF

"Triangle" Credit:
© 2017, Public. CC0 1.0 Universal
- Marco Hutter (https://github.com/javagl/) for Everything
