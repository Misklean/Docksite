// Let's create our own strategy
var strat = {};
var isTrend = false;

// Prepare everything our strat needs
strat.init = function() {
  var customEMAonesettings = {
    optInTimePeriod: 10
  }

  var customEMAtwosettings = {
    optInTimePeriod: 20
  }

  // add the indicator to the strategy
  this.addTulipIndicator('myemaone', 'ema', customEMAonesettings);
  this.addTulipIndicator('myematwo', 'ema', customEMAtwosettings);
}

// What happens on every new candle?
strat.update = function(candle) {
  // your code!
}

// For debugging purposes.
strat.log = function() {
  // your code!
}

// Based on the newly calculated
// information, check if we should
// update or not.
strat.check = function(candle) {
  var resultemaone = this.tulipIndicators.myemaone.result;
  var resultematwo = this.tulipIndicators.myematwo.result;

  if (!isTrend & resultemaone.result > resultematwo.result)
  {
    this.advice({
      direction: 'long', // or short
      trigger: { // ignored when direction is not "long"
        type: 'trailingStop',
        trailPercentage: 5
      }
    });
  }

  isTrend = resultemaone.result > resultematwo.result;
}

// Optional for executing code
// after completion of a backtest.
// This block will not execute in
// live use as a live gekko is
// never ending.
strat.end = function() {
  // your code!
}

module.exports = strat;

// open: 38676.55,
// high: 38713.8,
// low: 38676.55,
// close: 38712.82,
// vwp: 38708.36612843597,
// volume: 23.577810000000014,
// trades: 500
