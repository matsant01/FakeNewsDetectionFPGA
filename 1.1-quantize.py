'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''
import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

#from dataset_utils import input_fn_test, input_fn_quant

DIVIDER = '-----------------------------------------'



def quant_model(float_model,quant_model,batchsize,tfrec_dir,evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    sent_length=20
    voc_size=5000
    # calib_corpus is a part of the pre-processed initial dataset
    calib_corpus=['hous dem aid even see comey letter jason chaffetz tweet', 'flynn hillari clinton big woman campu breitbart', 'truth might get fire', 'civilian kill singl us airstrik identifi', 'iranian woman jail fiction unpublish stori woman stone death adulteri', 'jacki mason hollywood would love trump bomb north korea lack tran bathroom exclus video breitbart', 'beno hamon win french socialist parti presidenti nomin new york time', 'back channel plan ukrain russia courtesi trump associ new york time', 'obama organ action partner soro link indivis disrupt trump agenda', 'bbc comedi sketch real housew isi caus outrag', 'russian research discov secret nazi militari base treasur hunter arctic photo', 'us offici see link trump russia', 'ye paid govern troll social media blog forum websit', 'major leagu soccer argentin find home success new york time', 'well fargo chief abruptli step new york time', 'anonym donor pay million releas everyon arrest dakota access pipelin', 'fbi close hillari', 'chuck todd buzzfe donald trump polit favor breitbart', 'monica lewinski clinton sex scandal set american crime stori', 'rob reiner trump mental unstabl breitbart', 'abort pill order rise latin american nation zika alert new york time', 'nuke un histor treati ban nuclear weapon', 'exclus islam state support vow shake west follow manchest terrorist massacr breitbart', 'humili hillari tri hide camera caught min ralli', 'andrea tantaro fox news claim retali sex harass complaint new york time', 'hillari clinton becam hawk new york time', 'chuck todd buzzfe eic publish fake news breitbart', 'bori johnson brexit leader fumbl new york time', 'texa oil field rebound price lull job left behind new york time', 'bayer deal monsanto follow agribusi trend rais worri farmer new york time', 'russia move ban jehovah wit extremist new york time', 'still danger zone januari th', 'open thread u elect', 'democrat gutierrez blame chicago gun violenc nra breitbart', 'avoid peanut avoid allergi bad strategi new york time', 'mri show detail imag week unborn babi breitbart', 'best kind milk dairi', 'ryan locht drop speedo usa retail new york time', 'conserv urg session clean obama civil right divis breitbart', 'intern inquiri seal fate roger ail fox new york time', 'press tv debat duff lebanon hezbollah aoun presid video', 'samsung combust galaxi note unveil new smartphon new york time', 'poland vow referendum migrant quota amidst eu pressur public voic heard breitbart', 'spark inner revolut', 'studi half car crash involv driver distract cell phone breitbart', 'trump elect spark individu collect heal', 'ep fade black jimmi church w laura eisenhow restor balanc video', 'cognit true islam book review', 'donald trump win elect biggest miracl us polit histori', 'mind eat way fight bing new york time', 'major potenti impact corpor tax overhaul new york time', 'wonder glp like day elect', 'maker world smallest machin award nobel prize chemistri new york time', 'massiv anti trump protest union squar nyc live stream', 'review lion bring tear lost boy wipe dri googl new york time', 'u gener islam state chemic attack impact u forc', 'juri find oregon standoff defend guilti feder conspiraci gun charg', 'clinton campaign stun fbi reportedli reopen probe hillari clinton email', 'penc speak anti abort ralli new york time', 'berni sander say media trump gutless polit coward', 'make briquett daili wast', 'treason nyt vow reded report', 'dress like woman mean new york time', 'ella brennan still feed lead new orlean new york time', 'press asia agenda obama tread lightli human right new york time', 'democrat percent chanc retak senat new york time', 'judg spank transgend obsess obama lie redflag news', 'u diplomat urg strike assad syria new york time', 'franken call independ investig trump putin crush breitbart', 'louisiana simon bile u presidenti race tuesday even brief new york time', 'turkey threaten open migrant land passag europ row dutch', 'huma weiner dog hillari', 'colin kaepernick start black panther inspir youth camp wow', 'trump immigr polici explain new york time', 'mari tyler moor mourn dick van dyke star new york time', 'poison', 'trump fan ralli across nation support presid new york time', 'fox biz report help bash clinton ralli cover trump pack event day', 'fiction podcast worth listen new york time', 'mike birbiglia tip make small hollywood anywher new york time', 'invest strategist forecast collaps timelin last gasp econom cycl come', 'venezuela muzzl legislatur move closer one man rule new york time', 'whether john mccain mitt romney donald trump democrat alway run war women tactic destroy republican candid', 'breitbart news daili trump boom breitbart', 'white hous confirm gitmo transfer obama leav offic', 'poll voter heard democrat elect candid breitbart', 'migrant confront judgment day old deport order new york time', 'n u yale su retir plan fee new york time', 'technocraci real reason un want control internet', 'american driver regain appetit ga guzzler new york time', 'hillari clinton build million war chest doubl donald trump new york time', 'trump catch sick report snuck interview priceless respons', 'senat contact russian govern week', 'imag perfectli illustr struggl dakota access pipelin', 'washington state take refuge muslim rest countri take refuge muslim breitbart', 'ncaa big keep watch eye texa bathroom bill breitbart', 'massiv espn financi subscrib loss drag disney first quarter sale breitbart', 'megyn kelli contract set expir next year prime big show new york time', 'teacher suspend allow student hit trump pinata cinco de mayo', 'break trump express concern anthoni weiner illeg access classifi info month ago truthfe', 'snap share leap debut investor doubt valu vanish new york time', 'clinton campaign chair dinner top doj offici one day hillari benghazi hear', 'tv seri first femal mlb pitcher can one low rate season breitbart', 'seek best fit women final four return friday sunday new york time', 'propos canadian nation bird ruffl feather new york time', 'review beyonc make lemonad marit strife new york time', 'trump ask sharp increas militari spend offici say new york time', 'waterg smoke gun email discuss clean obama hillari email', 'chapo trap hous new left wing podcast flagrant rip right stuff', 'taiwan respond china send carrier taiwan strait new york time', 'mother octob surpris hous card come tumbl', 'explos assang pilger interview us elect expect riot hillari win', 'telescop ate astronomi track surpass hubbl new york time', 'close afghan pakistani border becom humanitarian crisi new york time', 'tv anchor arriv white hous lunch donald trump breitbart', 'pelosi republican tell trump bring dishonor presid breitbart', 'beauti prehistor world earth wasteland', 'ignor trump news week learn new york time', 'donald trump unveil plan famili bid women vote new york time', 'montana democrat vote bill ban sharia law call repugn breitbart', 'monsanto tribun go happen', 'offici simon bile world best gymnast new york time', 'liter hurt brain read econom idioci emit trumpkin libertarian', 'u n secretari gener complain mass reject global favor nation', 'trump bollywood ad meant sway indian american voter hilari fail video', 'fbi find previous unseen hillari clinton email weiner laptop', 'year american journalist kill conspiraci theori syria proven fact', 'report illeg alien forego food stamp stay trump radar', 'make netherland great hahaha spread worldwid', 'four kill injur jerusalem truck ram terror attack', 'leader salut comrad newt brutal megyn sic kelli beatdown play game', 'student black colleg got beaten mace protest kkk david duke', 'despit strict gun control one child youth shot everi day ontario', 'rise internet fan bulli new york time', 'newli vibrant washington fear trump drain cultur new york time', 'fed hold interest rate steadi plan slower increas new york time', 'battl unesco', 'latest test white hous pull easter egg roll new york time', 'burlesqu dancer fire investig secret servic trump assassin tweet breitbart', 'clinton haiti', 'cuomo christi parallel path top troubl got new york time', 'top place world allow visit', 'new studi link fluorid consumpt hypothyroid weight gain wors', 'jame matti secretari offens', 'black church burn spray paint vote trump', 'sear agre sell craftsman stanley black amp decker rais cash new york time', 'takata chief execut resign financi pressur mount new york time', 'goodby good black sabbath new york time', 'teen geisha doll gang bust arm robberi breitbart', 'mohamad khwei anoth virginia man palestinian american muslim charg terror', 'price obamacar replac nobodi wors financi breitbart', 'va fail properli examin thousand veteran', 'trump famili alreadi sworn secreci fake moon land soon', 'sport writer nfl great jim brown decad civil right work eras say nice thing donald trump breitbart', 'watch tv excus republican skip donald trump convent new york time', 'open letter trump voter told like', 'comment power corpor lobbi quietli back hillari nobodi talk runsinquicksand', 'hijack end peac libyan airlin land malta new york time', 'like girl girl geniu new york time', 'scientist say canadian bacteria fossil may earth oldest new york time', 'pro govern forc advanc syria amid talk u russia cooper new york time', 'cancer agenc fire withhold carcinogen glyphos document', 'work walk minut work new york time', 'steve harvey talk hous presid elect trump new york time', 'coalit u troop fight mosul offens come fire', 'uk citizen war hero get cheap pre fab hous muslim colon get taxpay fund luxuri council home', 'vet fight war fed demand money back illeg refuge keep money', 'mr trump wild ride new york time', 'fbi director comey bamboozl doj congress clinton', 'food natur unclog arteri prevent heart attack', 'death two state solut', 'comment parent date asleep car cop arriv kill facespac', 'donald trump team show sign post elect moder new york time', 'miami beach tri tame raucou street fishbowl drink stay new york time', 'doctor mysteri found dead summit breakthrough cure cancer', 'donald trump unsink candid new york time', 'shock new mock hillari ad campaign warn take us war enlistforh fightforh dieforh', 'exclus famili slain border patrol agent brian terri say eric holder among real crimin respons', 'trump tell report wall work ask israel breitbart', 'america surviv next year', 'commission start press cleveland indian logo new york time', 'un plan implant everyon biometr id drill', 'trump attack senat credibl gorsuch comment new york time', 'clinton advisor lose leak email hillari illeg activ', 'art laffer paul ryan perfect right breitbart', 'donald trump blame econom crash', 'pokemon go player inadvert stop peopl commit suicid japan', 'california senat race tale divers flail g p new york time', 'exclus sourc say megyn kelli would welcom back fox news', 'break preced obama envoy deni extens past inaugur day new york time', 'brexit vote go monti python may offer clue new york time', 'blind mystic predict bad news trump', 'total vet fail left wing snowden fan girl realiti winner get access nsa secret', 'somalia u escal shadow war new york time', 'free care bless victim orlando nightclub attack new york time', 'durabl democrat counti countri could go trump', 'fed challeng rais rate may existenti new york time', 'russia intent attack anyon absurd say vladimir putin', 'f investig errant flight involv harrison ford new york time', 'fed rais key interest rate cite strengthen economi new york time']
    onehot_calib=[one_hot(words,voc_size)for words in calib_corpus]
    embedded_calib=pad_sequences(onehot_calib,padding='pre',maxlen=sent_length)

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model)
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]

    # make TFRecord dataset and image processing pipeline
    #quant_dataset = input_fn_quant(tfrec_dir, batchsize, height, width)
    quant_dataset = embedded_calib
    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    if (evaluate):
        '''
        Evaluate quantized model
        '''
        #print('\n'+DIVIDER)
        #print ('Evaluating quantized model..')
        #print(DIVIDER+'\n')

        #test_dataset = input_fn_test(tfrec_dir, batchsize, height, width)

        #quantized_model.compile(optimizer=Adam(),
        #                        loss='sparse_categorical_crossentropy',
        #                        metrics=['accuracy'])

        #scores = quantized_model.evaluate(test_dataset,
        #                                  verbose=0)

        #print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        #print('\n'+DIVIDER)

    return



def main():


    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='float_model.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='quantized_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=20,                       help='Batchsize for quantization. Default is 50')
    ap.add_argument('-tfdir', '--tfrec_dir',type=str, default='build/tfrecords',              help='Full path to folder containing TFRecord files. Default is build/tfrecords')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --tfrec_dir    : ', args.tfrec_dir)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.tfrec_dir, args.evaluate)


if __name__ ==  "__main__":
    main()
