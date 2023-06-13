# AML_Final_Project
Final project for AppML at KU, by Jonas, Mikkel and Odi


Here is the data: http://www.slakh.com/

a better readme and explaination will come after the exams, but the general jist is that DNN random and DNN leastsquares are two ways of training a nextwork for doing music extraction. The notiable things in that is the directory, which is where you saved the track (i.e slakh's train, test or validation folder).

the lightgbm_tree is for doing the classification of what is playing. we determine that by if it is above or below 60 db. 

These are the main files, data generator is used to generate the data and is something we wrote.
the other file in is for prototyping, and will probably only work on babyslakh.

If you want to use babyslakh you should make sure your format is .wav and the sample_rate or sr is 16000 instead of 44100.

We tried to remove all magic values but have not completed it in total. so if you run the code there might come errors for not being on the same pc.
