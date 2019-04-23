#!/bin/bash

## ES-EN
#python unsupervised.py --src_lang es --tgt_lang en --src_emb wiki.es.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name spanish
#
## FR-EN
#python unsupervised.py --src_lang fr --tgt_lang en --src_emb wiki.fr.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name french
#
## ZH-EN
#python unsupervised.py --src_lang zh --tgt_lang en --src_emb wiki.zh.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name chinese
#
## KO-EN
#python unsupervised.py --src_lang ko --tgt_lang en --src_emb wiki.ko.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name korean
# JA-EN
#python unsupervised.py --src_lang ja --tgt_lang en --src_emb wiki.ja.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name korean

#python supervised.py --src_lang ja --tgt_lang en --src_emb wiki.ja.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name japanese_supervised
#python supervised.py --src_lang ko --tgt_lang en --src_emb wiki.ko.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name korean_supervised
#python supervised.py --src_lang zh --tgt_lang en --src_emb wiki.zh.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name chinese_supervised
#python supervised.py --src_lang fr --tgt_lang en --src_emb wiki.fr.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name french_supervised

# EN-FR
#python unsupervised.py --src_lang en --tgt_lang fr --src_emb wiki.en.vec --tgt_emb wiki.fr.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_french
#python supervised.py --src_lang en --tgt_lang fr --src_emb wiki.en.vec --tgt_emb wiki.fr.vec --n_refinement 8 --dico_train default --exp_name english_french_supervised
#
## PT-EN
#python unsupervised.py --src_lang pt --tgt_lang en --src_emb wiki.pt.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name portuguese
#python supervised.py --src_lang pt --tgt_lang en --src_emb wiki.pt.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name portuguese_supervised
#
## EN-PT
#python unsupervised.py --src_lang en --tgt_lang pt --src_emb wiki.en.vec --tgt_emb wiki.pt.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_portuguese
#python supervised.py --src_lang en --tgt_lang pt --src_emb wiki.en.vec --tgt_emb wiki.pt.vec --n_refinement 8 --dico_train default --exp_name english_portuguese_supervised
#
## EN-HE
#python unsupervised.py --src_lang en --tgt_lang he --src_emb wiki.en.vec --tgt_emb wiki.he.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_hebrew
#python supervised.py --src_lang en --tgt_lang he --src_emb wiki.en.vec --tgt_emb wiki.he.vec --n_refinement 8 --dico_train default --exp_name english_hebrew_supervised
#
## HE-EN
#python unsupervised.py --src_lang he --tgt_lang en --src_emb wiki.he.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name hebrew
#python supervised.py --src_lang he --tgt_lang en --src_emb wiki.he.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name hebrew_supervised
#
## EN-HI
#python unsupervised.py --src_lang en --tgt_lang hi --src_emb wiki.en.vec --tgt_emb wiki.hi.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_hindi
#python supervised.py --src_lang en --tgt_lang hi --src_emb wiki.en.vec --tgt_emb wiki.hi.vec --n_refinement 8 --dico_train default --exp_name english_hindi_supervised
#
## HI-EN
#python unsupervised.py --src_lang hi --tgt_lang en --src_emb wiki.hi.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name hindi
#python supervised.py --src_lang hi --tgt_lang en --src_emb wiki.hi.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name hindi_supervised
#
## EN-AR
#python unsupervised.py --src_lang en --tgt_lang ar --src_emb wiki.en.vec --tgt_emb wiki.ar.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_arabic
#python supervised.py --src_lang en --tgt_lang ar --src_emb wiki.en.vec --tgt_emb wiki.ar.vec --n_refinement 8 --dico_train default --exp_name english_arabic_supervised
#
## AR-EN
#python unsupervised.py --src_lang ar --tgt_lang en --src_emb wiki.ar.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name arabic
#python supervised.py --src_lang ar --tgt_lang en --src_emb wiki.ar.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name arabic_supervised
#
## EN-TH
#python unsupervised.py --src_lang en --tgt_lang th --src_emb wiki.en.vec --tgt_emb wiki.th.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_thai
#python supervised.py --src_lang en --tgt_lang th --src_emb wiki.en.vec --tgt_emb wiki.th.vec --n_refinement 8 --dico_train default --exp_name english_thai_supervised
#
## TH-EN
#python unsupervised.py --src_lang th --tgt_lang en --src_emb wiki.th.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name thai
#python supervised.py --src_lang th --tgt_lang en --src_emb wiki.ar.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name thai_supervised

## ES-EN
#python supervised.py --src_lang es --tgt_lang en --src_emb wiki.es.vec --tgt_emb wiki.en.vec --n_refinement 8 --dico_train default --exp_name spanish_supervised
#
## EN-KO
#python supervised.py --src_lang en --tgt_lang ko --src_emb wiki.en.vec --tgt_emb wiki.ko.vec --n_refinement 8 --dico_train default --exp_name english_korean_supervised
#
## EN-JA
#python unsupervised.py --src_lang en --tgt_lang ja --src_emb wiki.en.vec --tgt_emb wiki.ja.vec --n_refinement 8 --n_epochs 10 --epoch_size 250000 --normalize_embeddings center --lr_shrink 0.75 --exp_name english_japanese
#python supervised.py --src_lang en --tgt_lang ja --src_emb wiki.en.vec --tgt_emb wiki.ja.vec --n_refinement 8 --dico_train default --exp_name english_japanese_supervised

# JA-EN debug
python unsupervised.py --src_lang ja --tgt_lang en --src_emb wiki.ja.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 12 --epoch_size 150000 --normalize_embeddings center --lr_shrink 0.75 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.2 --exp_name japanese_debug

# EN-JA 150000, 0.2 repro w/4242
python unsupervised.py --src_lang en --tgt_lang ja --src_emb wiki.en.vec --tgt_emb wiki.ja.vec --n_refinement 8 --n_epochs 12 --epoch_size 150000 --normalize_embeddings center --lr_shrink 0.75 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.2 --exp_name english_japanese_debug

# JA-EN debug
python unsupervised.py --src_lang ja --tgt_lang en --src_emb wiki.ja.vec --tgt_emb wiki.en.vec --n_refinement 8 --n_epochs 15 --epoch_size 100000 --normalize_embeddings center --lr_shrink 0.8 --map_optimizer sgd,lr=0.2 --dis_optimizer sgd,lr=0.2 --exp_name japanese_debug

