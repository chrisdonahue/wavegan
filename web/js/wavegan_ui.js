window.wavegan = window.wavegan || {};

(function (deeplearn, wavegan) {
    // Config
    var cfg = wavegan.cfg;
    if (cfg.reqs.userCanceled) {
        document.getElementById('demo').setAttribute('hidden', '');
        document.getElementById('canceled').removeAttribute('hidden');
        return;
    }

    // Make a new random vector
    var random_vector = function (d) {
        var z = new Float32Array(d);
        for (var i = 0; i < d; ++i) {
            z[i] = (Math.random() * 2.) - 1.;
        }
        return z;
    };

    // Linear interpolation between two vectors
    var z_lerp = function (z0, z1, a) {
        if (z0.length !== z1.length) {
            throw 'Vector length differs';
        }

        var interp = new Float32Array(z0.length);
        for (var i = 0; i < z0.length; ++i) {
            interp[i] = (1. - a) * z0[i] + a * z1[i];
        }

        return interp;
    };

    var onResize = function (event) {
        var demo = document.getElementById('demo');
        var demoHeight = demo.offsetTop + demo.offsetHeight;
        var viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
        return;
    };

    var player = null;

    var onClick = function (event) {
        var z = [random_vector(wavegan.cfg.net.d_z), random_vector(wavegan.cfg.net.d_z)];
        var Gz = wavegan.net.eval(z);
        player.setSample(Gz[0], 16000);
        player.bang()
    };

    // Run once DOM loads
    var domReady = function () {
        cfg.debugMsg('DOM ready');

        var audioCtx = new window.AudioContext();

        var gainNode = audioCtx.createGain();
        gainNode.gain.value = 1.0;
        gainNode.connect(audioCtx.destination);

        player = new wavegan.player.ResamplingPlayer(audioCtx.sampleRate);

        var scriptProcessor = audioCtx.createScriptProcessor(512, 0, 1);
        console.log(scriptProcessor);
        scriptProcessor.onaudioprocess = function (event) {
            var output = event.outputBuffer;
            player.readBlock(output.getChannelData(0));
        };
        scriptProcessor.connect(gainNode);

        // (Gross) wait for net to be ready
        var wait = function() {
            if (wavegan.net.isReady()) {
                document.getElementById('overlay').setAttribute('hidden', '');
                document.getElementById('content').removeAttribute('hidden');
            }
            else {
                setTimeout(wait, 5);
            }
        };
        setTimeout(wait, 5);

        window.addEventListener('resize', onResize, true);
        onResize();

        var button = document.getElementById('sound');
        button.onclick = onClick;
    };

    // DOM load callbacks
    if (document.addEventListener) document.addEventListener("DOMContentLoaded", domReady, false);
    else if (document.attachEvent) document.attachEvent("onreadystatechange", domReady);
    else window.onload = domReady;

    // Exports
    wavegan.ui = {};

})(window.deeplearn, window.wavegan);
