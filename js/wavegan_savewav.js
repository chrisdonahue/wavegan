// Sourced from https://gist.github.com/asanoboy/3979747

window.wavegan = window.wavegan || {};

(function (wavegan) {
    var Wav = function(opt_params){
        this._sampleRate = opt_params && opt_params.sampleRate ? opt_params.sampleRate : 44100;
        this._channels = opt_params && opt_params.channels ? opt_params.channels : 2;  
        this._eof = true;
        this._bufferNeedle = 0;
        this._buffer;
    };

    Wav.prototype.setBuffer = function(buffer){
        this._buffer = this.getWavInt16Array(buffer);
        this._bufferNeedle = 0;
        this._internalBuffer = '';
        this._hasOutputHeader = false;
        this._eof = false;
    };

    Wav.prototype.getBuffer = function(len){
        var rt;
        if( this._bufferNeedle + len >= this._buffer.length ){
            rt = new Int16Array(this._buffer.length - this._bufferNeedle);
            this._eof = true;
        }
        else {
            rt = new Int16Array(len);
        }
        
        for(var i=0; i<rt.length; i++){
            rt[i] = this._buffer[i+this._bufferNeedle];
        }
        this._bufferNeedle += rt.length;
        
        return  rt.buffer;
    };

    Wav.prototype.eof = function(){
        return this._eof;
    };

    Wav.prototype.getWavInt16Array = function(buffer){
        var intBuffer = new Int16Array(buffer.length + 23), tmp;
        
        intBuffer[0] = 0x4952; // "RI"
        intBuffer[1] = 0x4646; // "FF"
        
        intBuffer[2] = (2*buffer.length + 15) & 0x0000ffff; // RIFF size
        intBuffer[3] = ((2*buffer.length + 15) & 0xffff0000) >> 16; // RIFF size
        
        intBuffer[4] = 0x4157; // "WA"
        intBuffer[5] = 0x4556; // "VE"
            
        intBuffer[6] = 0x6d66; // "fm"
        intBuffer[7] = 0x2074; // "t "
            
        intBuffer[8] = 0x0012; // fmt chunksize: 18
        intBuffer[9] = 0x0000; //
            
        intBuffer[10] = 0x0001; // format tag : 1 
        intBuffer[11] = this._channels; // channels: 2
        
        intBuffer[12] = this._sampleRate & 0x0000ffff; // sample per sec
        intBuffer[13] = (this._sampleRate & 0xffff0000) >> 16; // sample per sec
        
        intBuffer[14] = (2*this._channels*this._sampleRate) & 0x0000ffff; // byte per sec
        intBuffer[15] = ((2*this._channels*this._sampleRate) & 0xffff0000) >> 16; // byte per sec
        
        intBuffer[16] = 0x0004; // block align
        intBuffer[17] = 0x0010; // bit per sample
        intBuffer[18] = 0x0000; // cb size
        intBuffer[19] = 0x6164; // "da"
        intBuffer[20] = 0x6174; // "ta"
        intBuffer[21] = (2*buffer.length) & 0x0000ffff; // data size[byte]
        intBuffer[22] = ((2*buffer.length) & 0xffff0000) >> 16; // data size[byte]    

        for (var i = 0; i < buffer.length; i++) {
            tmp = buffer[i];
            if (tmp >= 1) {
                intBuffer[i+23] = (1 << 15) - 1;
            }
            else if (tmp <= -1) {
                intBuffer[i+23] = -(1 << 15);
            }
            else {
                intBuffer[i+23] = Math.round(tmp * (1 << 15));
            }
        }
        
        return intBuffer;
    };

    wavegan.savewav = {};
    wavegan.savewav.randomFilename = function () {
        return Math.random().toString(36).substring(7) + '.wav';
    };
    wavegan.savewav.saveWav = function (fn, buffer) {
        var wav = new Wav({sampleRate: 16000, channels: 1});
        wav.setBuffer(buffer);

        // Create file
        var srclist = [];
        while (!wav.eof()) {
            srclist.push(wav.getBuffer(1024));
        }
        var b = new Blob(srclist, {type:'audio/wav'});

        // Download
        var URLObject = window.webkitURL || window.URL;
        var url = URLObject.createObjectURL(b);
        var a = document.createElement('a');
        a.style = 'display:none';
        a.href = url;
        a.download = fn;
        a.click();
        URLObject.revokeObjectURL(url);
    };

})(window.wavegan);
