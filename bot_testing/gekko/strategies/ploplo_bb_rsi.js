// Let's create our own strategy
var strat = {};

// Prepare everything our strat needs
strat.init = function() {
  var customBBSettings = {
    optInTimePeriod: 30,
    optInNbStdDevs: 2

  }

  var customRSIsettings = {
    optInTimePeriod: 13
  }
  // add the indicator to the strategy
  this.addTulipIndicator('mybb', 'bbands', customBBSettings);
  this.addTulipIndicator('myrsi', 'rsi', customRSIsettings);
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
    var resultrsi = this.tulipIndicators.myrsi.result;

    // console.log(candle.start.format(), resultbb, resultrsi)
    if (candle.low < resultbb.bbandsLower && resultrsi.result < 25)
    {
      this.advice({
        direction: 'long',
        trigger: {
          type: 'trailingStop',
          trailPercentage: 2
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
