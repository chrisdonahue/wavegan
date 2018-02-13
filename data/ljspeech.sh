python make_tfrecord.py \
	/media/cdonahue/BIGBOYSBACK/datasets/ljspeech/train \
	/media/cdonahue/BIGBOYSBACK/datasets/dapper/ljspeech \
	--name train \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1.5 \

python make_tfrecord.py \
	/media/cdonahue/BIGBOYSBACK/datasets/ljspeech/valid \
	/media/cdonahue/BIGBOYSBACK/datasets/dapper/ljspeech \
	--name valid \
	--ext wav \
	--fs 16000 \
	--nshards 16 \
	--slice_len 1.5 \

python make_tfrecord.py \
	/media/cdonahue/BIGBOYSBACK/datasets/ljspeech/test \
	/media/cdonahue/BIGBOYSBACK/datasets/dapper/ljspeech \
	--name test \
	--ext wav \
	--fs 16000 \
	--nshards 16 \
	--slice_len 1.5 \
