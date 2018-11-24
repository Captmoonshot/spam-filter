import numpy as np
from preprocessing import make_bag, to_one_hot, oha_to_text, clean_line

import pickle

clf = pickle.load(open('data/pickles/classifier.pkl', 'rb'))
bow = pickle.load(open('data/pickles/bow.pkl', 'rb'))


def predict(txt):
	txt 				= clean_line(txt)
	oha_txt 			= to_one_hot(txt, add_to_bag=False, bow=bow)
	prediction_array 	= np.array(oha_txt)

	return clf.predict([prediction_array]) # 1 or 0

#print("\nSome predictions:")

# print(predict("what a bad one"))	# 0

# print(predict("how cool is that"))	# 1

# print(predict("just so good"))

# print(predict("that was just an incredible movie"))
"""
print(predict("You have won a guaranteed 32000 award or maybe even £1000 cash to claim ur award call free on 0800 ..... (18+). Its a legitimat efreefone number wat do u think???"))

print(predict("PRIVATE! Your 2003 Account Statement for 07808247860 shows 800 un-redeemed S. I. M. points. Call 08719899229 Identifier Code: 40411 Expires 06/11/04"))

print(predict("Want explicit SEX in 30 secs? Ring 02073162414 now! Costs 20p/min Gsex POBOX 2667 WC1N 3XX"))

print(predict("ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS. INDIA CUST SERVs SED YES. L8ER GOT MEGA BILL. 3 DONT GIV A SHIT. BAILIFF DUE IN DAYS. I O £250 3 WANT £800"))

print(predict("URGENT We are trying to contact you Last weekends draw shows u have won a £1000 prize GUARANTEED Call 09064017295 Claim code K52 Valid 12hrs 150p pm"))

print(predict("A £400 XMAS REWARD IS WAITING FOR YOU! Our computer has randomly picked you from our loyal mobile customers to receive a £400 reward. Just call 09066380611"))
"""






