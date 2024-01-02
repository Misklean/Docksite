// Let's create our own strategy
var strat = {};
var reverseTrend = false;

// Prepare everything our strat needs
strat.init = function() {
  var customMACDSettings = {
    optInFastPeriod: 12,
    optInSlowPeriod: 26,
    optInSignalPeriod: 9
  }

  var customEMAsettings = {
    optInTimePeriod: 200
  }
  // add the indicator to the strategy
  this.addTulipIndicator('mymacd', 'macd', customMACDSettings);
  this.addTulipIndicator('myema', 'ema', customEMAsettings);
  // rajouter un EMA pour 200 jours
  // long if macd crosses, but below the line + price au dessus du ema
  // trailing stop, ou short if 
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
    var resultmacd = this.tulipIndicators.mymacd.result;
    var resultema = this.tulipIndicators.myema.result;

    // console.log(candle.start.format(), resultmacd, resultema)
    if (resultema.result > candle.low && ((reverseTrend === false && resultmacd.macdHistogram > 0) && resultmacd.macd < 0))
    {
      this.advice({
        direction: 'long', // or short
        trigger: { // ignored when direction is not "long"
          type: 'trailingStop',
          trailPercentage: 2
        }
      });
    }

    reverseTrend = resultmacd.macdHistogram > 0;
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
