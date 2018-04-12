window.wavegan = window.wavegan || {};

(function (wavegan) {
    // Config
    var cfg = wavegan.cfg;

    var Sequencer = function (canvas, voices) {
        this.canvas = canvas;
        this.canvasCtx = this.canvas.getContext('2d');
        this.canvasWidth = this.canvas.width;
        this.canvasHeight = this.canvas.height;

        this.voices = voices;

        this.linesBuffer = document.createElement('canvas');
        this.linesBufferCtx = this.linesBuffer.getContext('2d');
        this.linesBuffer.width = this.canvasWidth;
        this.linesBuffer.height = this.canvasHeight;

        this.cellsBuffer = document.createElement('canvas');
        this.cellsBufferCtx = this.cellsBuffer.getContext('2d');
        this.cellsBuffer.width = this.canvasWidth;
        this.cellsBuffer.height = this.canvasHeight;

        // Create grid
        this.numCols = cfg.sequencer.numCols;
        this.numRows = this.voices.length;
        this.grid = [];
        for (var j = 0; j < this.numRows; ++j) {
            var row = [];
            for (var i = 0; i < this.numCols; ++i) {
                row.push(0);
            }
            this.grid.push(row);
        }
        this.grid[4][3] = 1;

        // Render
        this._redrawLines();
        this._redrawCells();

        // Bind click event
        var that = this;
        this.canvas.addEventListener('click', function (event) {
            var x = event.offsetX;
            var y = event.offsetY;
            var grid = that._absToGrid(x, y);

            var gridValid = true;
            gridValid &= grid.i >= 0;
            gridValid &= grid.i < that.numCols;
            gridValid &= grid.j >= 0;
            gridValid &= grid.j < that.numRows;

            if (gridValid) {
                var i = Math.floor(grid.i);
                var j = Math.floor(grid.j);
                that.grid[j][i] = 1 - that.grid[j][i];
                that._redrawCells();
                that.render();
            }
        });

        // Playback state
        this.delayMs = null;
        this.setTempoBpm(120);
        this.playing = false;
        this.tick = 0;
    };

    Sequencer.prototype._tick = function () {
        if (!this.playing) {
            return;
        }

        // Audio playback
        for (var j = 0; j < this.voices.length; ++j) {
            if (this.grid[j][this.tick] > 0) {
                this.voices[j].bang();
            }
        }

        // Render grid
        this.render();

        var that = this;
        setTimeout(function () {that._tick();}, this.delayMs);
        this.tick += 1;
        this.tick = this.tick % this.numCols;
    };
    Sequencer.prototype._absToGrid = function (x, y) {
        var labelWidth = cfg.sequencer.labelWidth;
        var gridWidth = this.canvasWidth - labelWidth;
        var gridHeight = this.canvasHeight;

        var cellWidth = gridWidth / this.numCols;
        var cellHeight = gridHeight / this.numRows;

        return {
            i: (x - labelWidth) / cellWidth,
            j: y / cellHeight
        };
    };
    Sequencer.prototype._gridToAbs = function (i, j) {
        var labelWidth = cfg.sequencer.labelWidth;
        var gridWidth = this.canvasWidth - labelWidth;
        var gridHeight = this.canvasHeight;

        var cellWidth = gridWidth / this.numCols;
        var cellHeight = gridHeight / this.numRows;

        return {
            x: (i * cellWidth) + labelWidth,
            y: j * cellHeight
        };
    };
    Sequencer.prototype._redrawCells = function () {
        var ctx = this.cellsBufferCtx;
        var w = this.canvasWidth;
        var h = this.canvasHeight;

        // Draw buffer
        ctx.clearRect(0, 0, w, h);
        var topLeft = this._gridToAbs(0, 0);
        var bottomRight = this._gridToAbs(this.numCols, this.numRows);

        // Draw grid
        ctx.fillStyle = '#ffffff';
        for (var j = 0; j < this.numRows; ++j) {
            for (var i = 0; i < this.numCols; ++i) {
                if (this.grid[j][i] > 0) {
                    var cellTopLeft = this._gridToAbs(i, j);
                    var cellBottomRight = this._gridToAbs(i + 1, j + 1);
                    var cellWidth = cellBottomRight.x - cellTopLeft.x;
                    var cellHeight = cellBottomRight.y - cellTopLeft.y;

                    ctx.fillRect(cellTopLeft.x, cellTopLeft.y, cellWidth, cellHeight);
                }
            }
        }

        // Draw grid lines
        ctx.drawImage(this.linesBuffer, 0, 0);
    };
    Sequencer.prototype._redrawLines = function () {
        var ctx = this.linesBufferCtx;
        var w = this.canvasWidth;
        var h = this.canvasHeight;

        // Clear background
        ctx.clearRect(0, 0, w, h);
        var topLeft = this._gridToAbs(0, 0);
        var bottomRight = this._gridToAbs(this.numCols, this.numRows);

        // Draw row lines
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        var rowStart = topLeft.x;
        var rowEnd = bottomRight.x;
        for (var j = 0; j < this.numRows + 1; ++j) {
            var y = this._gridToAbs(0, j).y;
            ctx.beginPath();
            ctx.moveTo(rowStart, y)
            ctx.lineTo(rowEnd, y);
            ctx.stroke();
        }

        // Draw columns
        var colStart = topLeft.y;
        var colEnd = bottomRight.y
        for (var i = 0; i < this.numCols + 1; ++i) {
            if (i % 4 == 0) {
                ctx.strokeStyle = '#ff0099';
                ctx.lineWidth = 2;
            }
            else {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
            }

            var x = this._gridToAbs(i, 0).x
            ctx.beginPath();
            ctx.moveTo(x, colStart)
            ctx.lineTo(x, colEnd);
            ctx.stroke();
        }
    };

    Sequencer.prototype.render = function () {
        var ctx = this.canvasCtx;
        var w = this.canvasWidth;
        var h = this.canvasHeight;

        // Draw background
        ctx.clearRect(0, 0, w, h);
        var topLeft = this._gridToAbs(0, 0);
        var bottomRight = this._gridToAbs(this.numCols, this.numRows);
        ctx.fillStyle = '#000000';
        ctx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

        // Draw cells
        ctx.drawImage(this.cellsBuffer, 0, 0);

        // Draw lines
        ctx.drawImage(this.linesBuffer, 0, 0);

        if (this.playing) {
            var topLeft = this._gridToAbs(this.tick, 0);
            var bottomRight = this._gridToAbs(this.tick + 1, this.numRows);
            ctx.fillStyle = '#ff0000';
            ctx.globalAlpha = 0.5;
            ctx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
            ctx.globalAlpha = 1;
        }
    };

    Sequencer.prototype.setTempoBpm = function (bpm) {
        var bps = bpm / 60;
        var cellsPerBeat = this.numCols / 4;
        var cellsPerSecond = bps * cellsPerBeat;
        var secondsPerCell = 1 / cellsPerSecond;
        this.delayMs = secondsPerCell * 1000;
    };
    Sequencer.prototype.play = function () {
        this.playing = true;
        this.tick = 0;
        this._tick();
    };
    Sequencer.prototype.stop = function () {
        this.playing = false;
        this.render();
    };
    Sequencer.prototype.clear = function () {
        for (var j = 0; j < this.numRows; ++j) {
            for (var i = 0; i < this.numCols; ++i) {
                this.grid[j][i] = 0;
            }
        }
        this._redrawCells();
        this.render();
    };

    // Exports
    wavegan.sequencer = {};
    wavegan.sequencer.Sequencer = Sequencer;

})(window.wavegan);
