# neural-style-transfer
Transfer the style of one image into another

### How this works

Our inputs are one content image and one style image. We will also initialise a generated image with random values in the shape of the content image.
We will train a convolutional neural network to optimize our cost function J(θ).

J(θ) = λ * J_content(C, G) + (1 - λ) * J_style(S, G)

Where:

λ is our parameter with a value between 0 and 1 which controls how much our generated 
image should look like the content image.

J_content(C, G) is the similarity between the content image C and the generated image G, which is defined as: J_content(C, G) = 1/2 * || a_c[l] - a_g[l] || ** 2,
a_c is the activation of the last layer in the input when you feed in the content image and a_g[l] is the activation of the last layer when you feed in the generated image. We multiply by one half so the derivative is easier when doing backprop.

J_style(S, G) is the similarity between the style image S and the generated image G, which is defined as: J_style(S, G) = || S - G || ** 2

