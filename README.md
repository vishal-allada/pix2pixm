# Modified pix2pix

Previously the upsampling part of generator contains only conv2dtranspose(size4, stride 2) which is responsible for increasing image size.

Now we will upsample using a upsample2d and conv2dtranspose(size3, stride 1) to refine feature maps.

- Data and artifacts folder link: https://drive.google.com/drive/u/0/folders/1tiQiYqeiZjDCxzdk7ZXJmCk_HjKjqv38
