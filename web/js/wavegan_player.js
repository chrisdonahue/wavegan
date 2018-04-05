window.wavegan = window.wavegan || {};

(function (wavegan) {
    // Config
    var cfg = wavegan.cfg;

    var ResamplingPlayer = function (fs) {
        this.sample = null;
        this.sampleIdx = 0;
        this.sampleIdxInc = 0;

        this.playing = false;
        this.fs = fs;
    };
    ResamplingPlayer.prototype.setSample = function (sample, sampleFs) {
        var samplePadded = new Float32Array(sample.length + 1);
        for (var i = 0; i < sample.length; ++i) {
            samplePadded[i] = sample[i];
        }
        samplePadded[i] = 0;

        this.sample = samplePadded;
        this.sampleLength = sample.length;
        this.sampleIdx = 0;
        this.sampleIdxInc = sampleFs / this.fs;
    };
    ResamplingPlayer.prototype.bang = function () {
        this.sampleIdx = 0;
        this.playing = true;
    };
    ResamplingPlayer.prototype.readBlock = function (buffer) {
        if (!this.playing) {
            return;
        }

        var sample = this.sample;
        var sampleLength = this.sampleLength;
        var sampleIdx = this.sampleIdx;
        var sampleIdxInc = this.sampleIdxInc;
        var floor, frac;
        for (var i = 0; i < buffer.length; ++i) {
            floor = Math.floor(sampleIdx);
            frac = sampleIdx - floor;

            if (floor < sampleLength) {
                buffer[i] += (1 - frac) * sample[floor] + frac * sample[floor + 1];
            }
            else {
                this.playing = false;
                break;
            }

            sampleIdx += sampleIdxInc;
        }

        this.sampleIdx = sampleIdx;
    };

    // Exports
    wavegan.player = {};
    wavegan.player.ResamplingPlayer = ResamplingPlayer;

})(window.wavegan);
