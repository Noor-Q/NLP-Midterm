# NLP-Midterm
This reprosetory contains 3 files for extracting keywords from Turkish language documents.
#Naive Bayes
This program uses Naive bayes and cross validation to find keywords, for this program to work correctly, the Training set must be placed in a folder titles "TrainingSet" and the testing Data must be in a folder "TestSet" . This program is rather slow and doesn't give good results.
#KeyBERT_Keywords
in this program the KeyBERT library is used to extract 4 keywords, 4 bigrams, and 4 trigrams then compare the results to the labled keywords and calculate the accuracy. For this program to work the testing Data must be in a folder "TestSet". Despite the fact that KeyBert represents the state of art keyword extraction tool, it does not support Turkish very well.Due to this this code is both slow and has low accuracy.
#Word_frequency_Keywords
This code uses simple NLTK word frequency tools to extract 4 keywords, 4 bigrams, and 4 trigrams then compare the results to the labled keywords and calculate the accuracy.For this program to work the testing Data must be in a folder "TestSet". This Code is the fastest to excute and produces somewhat better results compared to the other 2, but the accuracy is still low. 
