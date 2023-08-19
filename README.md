# **Clasificating Hand-Written Characters using Convolutional Neuronal Networks.**

Convolutional Neuronal Networks (_CNN_) are extensively employed in image classification. The following project acts as a basic classifier of Hand-Written characters ranging from 0 to 9 and from A to Z (including lowercase and capital letters). I used three different types of networks in this model:

+ A dense one (_Dense_)
+ One without DropOut (_CNN_)
+ One with DropOut (_CNN&DO_)

Here I present the script used for the _CNN&DO_ model

![CNN DO code](https://github.com/JPabl04/ClasificationCNN/assets/142553256/71d8a93e-4c36-4718-a94f-53102fadde8a)

Where one can see that the input images dimensions are 28x28 px in black and white scale. We apply a set of 32 and 64 convolutional filters in addition to a 2x2 MaxConvolution matrix, a 50% DropOut is used to avoid overfitting the model.

Below is presented a comparison between the three models and its respectively accuracy values

![gain(3)](https://github.com/JPabl04/ClasificationCNN/assets/142553256/c96af989-0925-422c-831f-1992dd45df4d)

It's easy to see that the _CNN&DO_ has a better performance than the _CNN_ does

![gain(2)](https://github.com/JPabl04/ClasificationCNN/assets/142553256/a439e667-7f32-4fdf-b8dc-bae6a8ac8713)

This shows that CNN are better than conventional Dense Neural Networks for picture classification. Furthermore, the continuation of this project could involve word classification alternatively.
