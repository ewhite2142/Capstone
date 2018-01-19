## **Classifying Mountains with Photos and Neural Networks**

This project attempts to classify mountains \(let's call them "summits"\) into certain categories, specifically \(i\) which state the summit is in, and \(ii\) whether the summit is a named a "Mount", "Mountain", or "Peak". 

As I am an avid hiker, I wondered if there were any convention to why some summits are called a mount \(e.g. "Mount Evans"\), a mountain \("Green Mountain"\), or a peak \("Pikes Peak"\). Apparently, there is no convention, although some research by a college student, Step Abegg, used elevation, prominence \(how much higher is the summit relative to its neighbors\), and isolation \(a measure of distance between summits\) in a multinomial logistic regression and found a pattern that works most of the time. The pattern is:

* Mountains - tend to be lower elevation and rounded
* Mounts - tend to be high elevation and prominent
* Peaks - tend  to be pointed with other peaks nearby.

My idea is to see if a convolutional neural network \("cnn"\), an advanced algorythm used for classifying images, can replicate the finding of the numeric regression model. While I'm at it, I though I'd try to see the the cnn could classify the photos in other ways, e.g. predict which state the summit is in.
