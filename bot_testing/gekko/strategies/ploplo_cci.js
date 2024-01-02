// Let's create our own strategy
var strat = {};

// Prepare everything our strat needs
strat.init = function() {
  var customCCIsettings = {
    optInTimePeriod: 10
  }

  // add the indicator to the strategy
  this.addTulipIndicator('mycci', 'cci', customCCIsettings);
}

// What happens on every new candle?
strat.update = function(candle) {
  
}

// For debugging purposes.
strat.log = function() {
  
}

// Based on the newly calculated
// information, check if we should
// update or not.
strat.check = function(candle) {
  var resultcci = this.tulipIndicators.mycci.result;

  // console.log(candle.start.format(), resultcci.result);
  console.log('[CANDLE]', candle.start.format('YYYY-MM-DD HH:mm:ss'), resultcci.result);

  if (resultcci.result < -100)
  {
    this.advice({
      direction: 'long', // or short
      trigger: { // ignored when direction is not "long"
        type: 'trailingStop',
        trailPercentage: 0.2
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
