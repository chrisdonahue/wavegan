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
    var random_vector = function () {
        var d = wavegan.cfg.net.zDim;
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

    // Class to handle UI interactions with player/visualizer
    var Zactor = function (fs, div) {
        this.canvas = div.children[0];
        this.button = div.children[1];
        this.player = new wavegan.player.ResamplingPlayer(fs);
        this.visualizer = new wavegan.visualizer.WaveformVisualizer(this.canvas);
        this.z = null;
        this.Gz = null;
        this.filename = null;

        var that = this;
        this.canvas.onclick = function (event) {
            that.bang();
        };

        // Change button
        div.children[1].onclick = function (event) {
            that.randomize();
        };

        // Save button
        div.children[2].onclick = function (event) {
            if (that.Gz !== null) {
                if (that.filename === null) {
                    that.filename = wavegan.savewav.randomFilename();
                }
                wavegan.savewav.saveWav(that.filename, that.Gz);
            }
        };
    };
    Zactor.prototype.setPrerendered = function (z, Gz) {
        this.z = z;
        this.Gz = Gz;
        this.filename = null;
        this.player.setSample(Gz, 16000);
        this.visualizer.setSample(Gz);
    };
    Zactor.prototype.setZ = function (z) {
        var Gz = wavegan.net.eval([z])[0];
        this.setPrerendered(z, Gz);
    };
    Zactor.prototype.randomize = function () {
        var z = random_vector();
        this.setZ(z);
    };
    Zactor.prototype.readBlock = function (buffer) {
        this.player.readBlock(buffer);
    };
    Zactor.prototype.bang = function () {
        this.player.bang();
    };

    // Initializer for waveform players/visualizers
    var zactors = null;
    var initZactors = function (audioCtx) {
        var nzactors = cfg.ui.zactorNumRows * cfg.ui.zactorNumCols;

        // Create zactors
        zactors = [];
        for (var i = 0; i < nzactors; ++i) {
            var div = document.getElementById('zactor' + String(i));
            zactors.push(new Zactor(audioCtx.sampleRate, div));
        }

        // Render initial batch
        var zs = [];
        for (var i = 0; i < nzactors; ++i) {
            zs.push(random_vector());
        }
        var Gzs = wavegan.net.eval(zs);
        for (var i = 0; i < nzactors; ++i) {
            zactors[i].setPrerendered(zs[i], Gzs[i]);
        }

        // Hook up audio
        var scriptProcessor = audioCtx.createScriptProcessor(512, 0, 1);
        scriptProcessor.onaudioprocess = function (event) {
            var buffer = event.outputBuffer.getChannelData(0);
            for (var i = 0; i < buffer.length; ++i) {
                buffer[i] = 0;
            }
            for (var i = 0; i < nzactors; ++i) {
                zactors[i].readBlock(buffer);
            }
        };

        return scriptProcessor;
    };

    // Global resize callback
    var onResize = function (event) {
        var demo = document.getElementById('demo');
        var demoHeight = demo.offsetTop + demo.offsetHeight;
        var viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
        return;
    };

    // Global keyboard callback
    var onKeydown = function (event) {
        var key = event.keyCode;
        var digit = key - 48;
        var zactorid = digit - 1;
        var shifted = event.getModifierState('Shift');
        if (zactorid >= 0 && zactorid < 8) {
            if (shifted) {
                zactors[zactorid].randomize();
            }
            else {
                zactors[zactorid].bang();
            }
        }
    };

    // Run once DOM loads
    var domReady = function () {
        cfg.debugMsg('DOM ready');

        // Create grid
        var cellTemplate = document.getElementById('zactor-template').innerHTML;
        var i = 0;
        var gridHtml = '';
        for (var j = 0; j < cfg.ui.zactorNumRows; ++j) {
            gridHtml += '<div class="row">';
            for (var k = 0; k < cfg.ui.zactorNumCols; ++k) {
                gridHtml += cellTemplate.replace('{ID}', 'zactor' + String(i));
                ++i;
            }
            gridHtml += '</div>';
        }
        document.getElementById('zactors').innerHTML = gridHtml;

        // Initialize audio
        var audioCtx = new window.AudioContext();
        var gainNode = audioCtx.createGain();
        gainNode.gain.value = 0.25;
        gainNode.connect(audioCtx.destination);

        // (Gross) wait for net to be ready
        var wait = function() {
            if (wavegan.net.isReady()) {
                var scriptProcessor = initZactors(audioCtx);
                scriptProcessor.connect(gainNode);
                document.getElementById('overlay').setAttribute('hidden', '');
                document.getElementById('content').removeAttribute('hidden');
            }
            else {
                setTimeout(wait, 5);
            }
        };
        setTimeout(wait, 5);

        // Global resize callback
        window.addEventListener('resize', onResize, true);
        onResize();

        // Global key listener callback
        window.addEventListener('keydown', onKeydown, true);
    };

    // DOM load callbacks
    if (document.addEventListener) document.addEventListener("DOMContentLoaded", domReady, false);
    else if (document.attachEvent) document.attachEvent("onreadystatechange", domReady);
    else window.onload = domReady;

    // Exports
    wavegan.ui = {};

})(window.deeplearn, window.wavegan);
