window.wavegan = window.wavegan || {};

(function (dl, wavegan) {
    // Config
    var cfg = wavegan.cfg;
    if (cfg.reqs.userCanceled) {
        return;
    }

    // Network state
    var net = {
        vars: null,
        ready: false
    };

    // Hardware state
    var hw = {
        math: null,
        ready: false
    };

    // Initialize hardware (uses WebGL if possible)
    var initHw = function (graph) {
        // TODO: update this
        try {
            new dl.NDArrayMathGPU();
            cfg.debugMsg('WebGL supported');
        }
        catch(err) {
            new dl.NDArrayMathCPU();
            cfg.debugMsg('WebGL not supported');
        }

        hw.math = dl.ENV.math;

        hw.ready = true;
        cfg.debugMsg('Hardware ready');
    };

    // Initialize network and hardware
    var initVars = function () {
        var varLoader = new dl.CheckpointLoader(cfg.net.ckptDir);
        varLoader.getAllVariables().then(function (vars) {
            net.vars = vars;
            net.ready = true;

            cfg.debugMsg('Variables loaded');
        });
    };

    // Exports
    wavegan.net = {};

    wavegan.net.isReady = function () {
        return net.ready && hw.ready;
    };

    wavegan.net.getCherries = function () {
        if (!wavegan.net.isReady()) {
            throw 'Hardware not ready';
        }
        if ('cherries' in net.vars) {
            var cherries = net.vars['cherries'];
            var _zs = [];
            for (var i = 0; i < cherries.shape[0]; ++i) {
                var _z = new Float32Array(cfg.net.zDim);
                for (var j = 0; j < cfg.net.zDim; ++j) {
                    _z[j] = cherries.get(i, j);
                }
                _zs.push(_z);
            }
            return _zs;
        }
        else {
            return null;
        }
    };

    wavegan.net.eval = function (_z) {
        if (!wavegan.net.isReady()) {
            throw 'Hardware not ready';
        }
        for (var i = 0; i < _z.length; ++i) {
            if (_z[i].length !== cfg.net.zDim) {
                throw 'Input shape incorrect'
            }
        }

        var m = hw.math;

        // Reshape input to 2D array
        var b = _z.length;
        var _z_flat = new Float32Array(b * cfg.net.zDim);
        for (var i = 0; i < b; ++i) {
            for (var j = 0; j < cfg.net.zDim; ++j) {
                _z_flat[i * cfg.net.zDim + j] = _z[i][j];
            }
        }
        var x = dl.Array2D.new([b, cfg.net.zDim], _z_flat);

        // Project to [b, 1, 16, 1024]
        x = m.matMul(x, net.vars['G/z_project/dense/kernel']);
        x = m.add(x, net.vars['G/z_project/dense/bias']);
        x = m.relu(x);
        x = x.reshape([b, 1, 16, 1024]);

        // Conv 0 to [b, 1, 64, 512]
        x = m.conv2dTranspose(x,
            net.vars['G/upconv_0/conv2d_transpose/kernel'],
            [b, 1, 64, 512],
            [1, 4],
            'same');
        x = m.add(x, net.vars['G/upconv_0/conv2d_transpose/bias']);
        x = m.relu(x);

        // Conv 1 to [b, 1, 256, 256]
        x = m.conv2dTranspose(x,
            net.vars['G/upconv_1/conv2d_transpose/kernel'],
            [b, 1, 256, 256],
            [1, 4],
            'same');
        x = m.add(x, net.vars['G/upconv_1/conv2d_transpose/bias']);
        x = m.relu(x);

        // Conv 2 to [b, 1, 1024, 128]
        x = m.conv2dTranspose(x,
            net.vars['G/upconv_2/conv2d_transpose/kernel'],
            [b, 1, 1024, 128],
            [1, 4],
            'same');
        x = m.add(x, net.vars['G/upconv_2/conv2d_transpose/bias']);
        x = m.relu(x);

        // Conv 3 to [b, 1, 4096, 64]
        x = m.conv2dTranspose(x,
            net.vars['G/upconv_3/conv2d_transpose/kernel'],
            [b, 1, 4096, 64],
            [1, 4],
            'same');
        x = m.add(x, net.vars['G/upconv_3/conv2d_transpose/bias']);
        x = m.relu(x);

        // Conv 4 to [b, 1, 16384, 1]
        x = m.conv2dTranspose(x,
            net.vars['G/upconv_4/conv2d_transpose/kernel'],
            [b, 1, 16384, 1],
            [1, 4],
            'same');
        x = m.add(x, net.vars['G/upconv_4/conv2d_transpose/bias']);
        x = m.tanh(x);

        // Post processing filter
        x = m.reshape(x, [b, 16384, 1]);
        if (cfg.net.ppFilt) {
            x = m.conv1d(x,
                net.vars['G/pp_filt/conv1d/kernel'],
                null,
                1,
                'same');
        }

        // Create Float32Arrays with result
        wavs = []
        for (var i = 0; i < b; ++i) {
            var wav = new Float32Array(16384);
            for (var j = 0; j < 16384; ++j) {
                wav[j] = x.get(i, j, 0);
            }
            wavs.push(wav);
        }

        return wavs
    };

    // Run immediately
    initVars();
    initHw();

})(window.dl, window.wavegan);
