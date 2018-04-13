window.wavegan = window.wavegan || {};

(function (deeplearn, wavegan) {
    var cfg = wavegan.cfg;

    // Prompt if no WebGL
    try {
        var math = new deeplearn.NDArrayMathGPU();
    }
    catch (err) {
        cfg.debugMsg('WebGL error: ' + String(err));

        if (confirm(cfg.reqs.noWebGlWarning) === false) {
            cfg.reqs.userCanceled = true;
            cfg.debugMsg('User canceled demo (no WebGL)');
        }

        document.getElementById('canceled').removeAttribute('hidden');
        document.getElementById('content').removeAttribute('hidden');
        document.getElementById('overlay').setAttribute('hidden', '');

        return;
    }

    // Prompt if mobile
    if(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ) {
        if (confirm(cfg.reqs.mobileWarning) === false) {
            cfg.reqs.userCanceled = true;
            cfg.debugMsg('User canceled demo (mobile)');
        }

        document.getElementById('canceled').removeAttribute('hidden');
        document.getElementById('content').removeAttribute('hidden');
        document.getElementById('overlay').setAttribute('hidden', '');

        return;
    }

})(window.deeplearn, window.wavegan);
