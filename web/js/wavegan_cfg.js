window.wavegan = window.wavegan || {};

(function (wavegan) {
    var debug = true;

    // Config
    wavegan.cfg = {
        reqs: {
            userCanceled: false,
            noWebGlWarning: 'Warning: We did not find WebGL in your browser. This demo uses WebGL to accelerate neural network computation. Performance will be slow and may hang your browser. Continue?',
            mobileWarning: 'Warning: This demo runs a neural network in your browser. It appears you are on a mobile device. Consider running the demo on your laptop instead. Continue?'
        },
        net: {
            ckptDir: 'ckpts/drums',
            ppFilt: true,
            zDim: 100,
            cherries: [0, 1, 2, 3, 4, 5, 6, 7]
        },
        audio: {
            gainDefault: 0.5,
            reverbDefault: 0.25,
            reverbLen: 2,
            reverbDecay: 10
        },
        ui: {
            canvasFlushDelayMs: 25,
            visualizerGain: 1,
            zactorNumRows: 2,
            zactorNumCols: 4,
            rmsAnimDelayMs: 25
        },
        sequencer: {
            labelWidth: 80,
            numCols: 16,
            tempoMin: 30,
            tempoMax: 300,
            tempoDefault: 120,
            swingMin: 0.5,
            swingMax: 0.8,
            swingDefault: 0.5
        }
    };

    wavegan.cfg.debugMsg = function (msg) {
        if (debug) {
            console.log(msg);
        }
    };

})(window.wavegan);
