window.wavegan = window.wavegan || {};

(function (wavegan) {
    // Config
    var cfg = wavegan.cfg;

    var WaveformVisualizer = function (canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.canvasCtx = this.canvas.getContext('2d');
    };
    WaveformVisualizer.prototype.setSample = function (sample, sampleFs) {
        var ctx = this.canvasCtx;

        ctx.clearRect(0, 0, 320, 180);
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(0, 0, 320, 180);

        var w = 320;
        var h = 180;
        var hd2 = h / 2;
        var t = sample.length;

        var pxdt = t / w;

        ctx.fillStyle = '#000000';

        for (var i = 0; i < w; ++i) {
            var tl = Math.floor(i * pxdt);
            var th = Math.floor((i + 1) * pxdt);

            var max = 0;
            for (var k = tl; k < th; ++k) {
                if (Math.abs(sample[k]) > max) {
                    max = Math.abs(sample[k]);
                }
            }

            var rect_height = max * hd2;

            ctx.fillRect(i, hd2 - rect_height, i+1, rect_height);
            ctx.fillRect(i, hd2, i+1, rect_height);
        }
    };

    // Exports
    wavegan.visualizer = {};
    wavegan.visualizer.WaveformVisualizer = WaveformVisualizer;

})(window.wavegan);
