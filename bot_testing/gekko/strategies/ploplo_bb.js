// Let's create our own strategy
var strat = {};

// Prepare everything our strat needs
strat.init = function() {
  var customBBSettings = {
    optInTimePeriod: 20,
    optInNbStdDevs: 2

  }
  // add the indicator to the strategy
  this.addTulipIndicator('mybb', 'bbands', customBBSettings);
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
  var resultbb = this.tulipIndicators.mybb.result;

  if (candle.close > resultbb.bbandsUpper)
  {
    this.advice({
      direction: 'long', // or short
      trigger: { // ignored when direction is not "long"
        type: 'trailingStop',
        trailPercentage: 5
      }
    });
  }
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
