# -*- coding: utf-8 -*-
import bagofwords

model = bagofwords.AnalyzeReview()
review = str()


print model.predict(review)
