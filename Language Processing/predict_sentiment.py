# -*- coding: utf-8 -*-
import bagofwords

model = bagofwords.AnalyzeReview()
review = str('Looking for a film which is so bad that it’s good? This is it then; Hind Ka Napak ko Jawab — MSG Lion Heart - 2, '
             'the third foray of the godman, Dr Gurmeet Ram Rahim Singh Insan aka Dr MSG, into the world of movies. On paper, '
             'he is an Indian spy called Sher-e-Hind, feared by all and hell-bent on taking revenge against Pakistan for all the terrorist attacks on India. '
             'On screen though, he is like a six-foot-tall baby, stumbling around in a crockery shop with an AK 47.')


print model.predict(review)
