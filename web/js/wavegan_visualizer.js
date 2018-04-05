window.wavegan = window.wavegan || {};

(function (wavegan) {
    // Config
    var cfg = wavegan.cfg;

    var WaveformVisualizer = function (canvas) {
        this.canvas = canvas;
        this.canvasCtx = this.canvas.getContext('2d');
        this.canvasWidth = this.canvas.width;
        this.canvasHeight = this.canvas.height;
    };
    WaveformVisualizer.prototype.setSample = function (sample) {
        var ctx = this.canvasCtx;
        var w = this.canvasWidth;
        var h = this.canvasHeight;
        var gain = cfg.ui.visualizerGain;

        var hd2 = h / 2;
        var t = sample.length;
        var pxdt = t / w;

        // Clear background
        ctx.clearRect(0, 0, 320, 180);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 320, 180);

        // Draw DC line
        ctx.fillStyle = '#33ccff';
        ctx.fillRect(0, hd2, w, 1);

        // Draw waveform
        ctx.fillStyle = '#33ccff';
        for (var i = 0; i < w; ++i) {
            var tl = Math.floor(i * pxdt);
            var th = Math.floor((i + 1) * pxdt);

            var max = 0;
            for (var k = tl; k < th; ++k) {
                if (Math.abs(sample[k]) > max) {
                    max = Math.abs(sample[k]);
                }
            }

            var rect_height = max * hd2 * gain;

            ctx.fillRect(i, hd2 - rect_height, 1, rect_height);
            ctx.fillRect(i, hd2, 1, rect_height);
        }
    };

    // Exports
    wavegan.visualizer = {};
    wavegan.visualizer.WaveformVisualizer = WaveformVisualizer;

})(window.wavegan);
