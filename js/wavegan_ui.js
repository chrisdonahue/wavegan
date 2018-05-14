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
    var globalAudioCtxChromeWorkaround = null;
    var globalAudioCtxHasBeenResumed = false;
    var Zactor = function (fs, div, name, color) {
        this.canvas = div.children[0];
        this.button = div.children[1];
        this.player = new wavegan.player.ResamplingPlayer(fs);
        this.visualizer = new wavegan.visualizer.WaveformVisualizer(this.canvas, name, color);
        this.animFramesRemaining = 0;
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
        var oldGain = gainNode.gain.value;
        gainNode.gain.value = 0;

        var z = random_vector();
        this.setZ(z);

        gainNode.gain.value = oldGain;
    };
    Zactor.prototype.readBlock = function (buffer) {
        this.player.readBlock(buffer);
    };
    Zactor.prototype.bang = function () {
        if (!globalAudioCtxHasBeenResumed && globalAudioCtxChromeWorkaround !== null) {
            globalAudioCtxChromeWorkaround.resume();
            globalAudioCtxHasBeenResumed = true;
        }

        this.player.bang();

        var animFramesTot = Math.round(1024 / cfg.ui.rmsAnimDelayMs);
        this.animFramesRemaining = animFramesTot;
        var lastRemaining = this.animFramesRemaining;
        var that = this;
        var animFrame = function () {
            var rms = that.player.getRmsAmplitude();
            var initPeriod = animFramesTot - that.animFramesRemaining;
            if (initPeriod < 8) {
                var fade = initPeriod / 8;
                rms = (1 - fade) * 0.25 + fade * rms;
            }
            that.visualizer.render(rms);

            if (that.animFramesRemaining > 0 && lastRemaining === that.animFramesRemaining) {
                --that.animFramesRemaining;
                --lastRemaining;
                setTimeout(animFrame, cfg.ui.rmsAnimDelayMs);
            }
        };

        animFrame();
    };

    // Initializer for waveform players/visualizers
    var zactors = null;
    var initZactors = function (audioCtx, cherries) {
        var nzactors = cfg.ui.zactorNumRows * cfg.ui.zactorNumCols;

        // Create zactors
        zactors = [];
        for (var i = 0; i < nzactors; ++i) {
            var div = document.getElementById('zactor' + String(i));
            var name = 'Drum ' + String(i + 1);
            var hue = (i / (nzactors - 1)) * 255;
            var hsl = 'hsl(' + String(hue) + ', 80%, 60%)';
            zactors.push(new Zactor(audioCtx.sampleRate, div, name, hsl));
        }

        // Render initial batch
        var zs = [];
        if (cherries === null || cfg.net.cherries.length != nzactors) {
            for (var i = 0; i < nzactors; ++i) {
                zs.push(random_vector());
            }
        }
        else {
            for (var i = 0; i < nzactors; ++i) {
                zs.push(cherries[cfg.net.cherries[i]]);
            }
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

    // Sequencer state
    var sequencer = null;

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

        // Space bar
        if (key == 32) {
            sequencer.toggle();
        }
    };

    var initSlider = function (sliderId, sliderMin, sliderMax, sliderDefault, callback) {
        var slider = document.getElementById(sliderId);
        slider.value = 10000 * ((sliderDefault - sliderMin) / (sliderMax - sliderMin));
        callback(sliderDefault);
        slider.addEventListener('input', function (event) {
            var valUi = slider.value / 10000;
            var val = (valUi * (sliderMax - sliderMin)) + sliderMin;
            callback(val);
        }, true);
    };

    var createReverb = function (audioCtx) {
        var sampleRate = audioCtx.sampleRate;
        var reverbLen = Math.floor(sampleRate * cfg.audio.reverbLen);
        var reverbDcy = cfg.audio.reverbDecay;
        var impulse = audioCtx.createBuffer(2, reverbLen, sampleRate);
        var impulseL = impulse.getChannelData(0);
        var impulseR = impulse.getChannelData(1);
        for (var i = 0; i < reverbLen; ++i) {
            impulseL[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / reverbLen, reverbDcy);
            impulseR[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / reverbLen, reverbDcy);
        }
        var reverbNode = audioCtx.createConvolver();
        reverbNode.buffer = impulse;
        return reverbNode
    };

    // Run once DOM loads
    var gainNode = null;
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
        globalAudioCtxChromeWorkaround = audioCtx;

        var reverbNode = createReverb(audioCtx);
        var wet = audioCtx.createGain();
        var dry = audioCtx.createGain();
        gainNode = audioCtx.createGain();
        reverbNode.connect(wet);
        wet.connect(gainNode);
        dry.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        // (Gross) wait for net to be ready
        var wait = function() {
            if (wavegan.net.isReady()) {
                var scriptProcessor = initZactors(audioCtx, wavegan.net.getCherries());
                scriptProcessor.connect(reverbNode);
                scriptProcessor.connect(dry);

                var seqCanvas = document.getElementById('sequencer-canvas');
                sequencer = new wavegan.sequencer.Sequencer(seqCanvas, zactors);
                sequencer.render();

                document.getElementById('overlay').setAttribute('hidden', '');
                document.getElementById('content').removeAttribute('hidden');
            }
            else {
                setTimeout(wait, 5);
            }
        };
        setTimeout(wait, 5);

        // Sequencer button callbacks
        document.getElementById('sequencer-play').addEventListener('click', function () {
            sequencer.play();
        });
        document.getElementById('sequencer-stop').addEventListener('click', function () {
            sequencer.stop();
        });
        document.getElementById('sequencer-clear').addEventListener('click', function () {
            sequencer.clear();
        });

        // Slider callbacks
        initSlider('gain',
                0, 1,
                cfg.audio.gainDefault,
                function (val) {
                    gainNode.gain.value = val * val * val * val;
        });
        initSlider('reverb',
                0, 1,
                cfg.audio.reverbDefault,
                function (val) {
                    dry.gain.value = (1 - val);
                    wet.gain.value = val;
        });
        initSlider('sequencer-tempo',
                cfg.sequencer.tempoMin, cfg.sequencer.tempoMax,
                cfg.sequencer.tempoDefault,
                function (val) {
                    if (sequencer !== null) {
                        sequencer.setTempoBpm(val);
                    }
        });
        initSlider('sequencer-swing',
                cfg.sequencer.swingMin, cfg.sequencer.swingMax,
                cfg.sequencer.swingDefault,
                function (val) {
                    if (sequencer !== null) {
                        sequencer.setSwing(val);
                    }
        });

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
