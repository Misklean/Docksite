// Let's create our own strategy
var strat = {};
var trendTreshold = 3;

// Prepare everything our strat needs
strat.init = function() {
  var customMACDsettings = {
    optInFastPeriod: 12,
    optInSlowPeriod: 26,
    optInSignalPeriod: 9
  }

  var customCCIsettings = {
    optInTimePeriod: 10
  }

  var customRSIsettings = {
    optInTimePeriod: 14
  }
  // add the indicator to the strategy
  this.addTulipIndicator('mymacd', 'macd', customMACDsettings);
  this.addTulipIndicator('mycci', 'cci', customCCIsettings);
  this.addTulipIndicator('myrsi', 'rsi', customRSIsettings);

  this.rsi_above = [];
  this.cci_above = [];
  this.macd_above = [];

  
}

// What happens on every new candle?
strat.update = function(candle) {
	this.macd_result = this.tulipIndicators.mymacd.result;
	this.cci_result = this.tulipIndicators.mycci.result;
	this.rsi_result = this.tulipIndicators.myrsi.result;
	this.new_candle = candle;

	if (this.macd_above.length >= trendTreshold)
	{
		this.macd_above.shift();
	}
	this.macd_above.push(this.macd_result.macdHistogram);

	if (this.rsi_above.length >= trendTreshold)
	{
		this.rsi_above.shift();
	}
	this.rsi_above.push(this.rsi_result.result);

	if (this.cci_above.length >= trendTreshold)
	{
		this.cci_above.shift();
	}
	this.cci_above.push(this.cci_result.result);
}

// For debugging purposes.
strat.log = function() {
    // console.log(this.new_candle.start.format(), this.macd_result, this.cci_result, this.rsi_result);
    // console.log(this.rsi_above)
    // console.log(this.cci_above)
    // console.log(this.macd_line_above)
    // console.log(this.macd_signal_above)
	// console.log(this.new_candle.start.format(), this.new_candle.close)
	console.log('[CANDLE]', this.new_candle.start.format('YYYY-MM-DD HH:mm:ss'), this.macd_result.macdHistogram, this.cci_result.result, this.rsi_result.result);
}

// Based on the newly calculated
// information, check if we should
// update or not.
strat.check = function(candle) {
	var macd_line = this.macd_result.macd;
	var macd_signal = this.macd_result.macdSignal;
	var rsi = this.rsi_result.result;
	var cci = this.cci_result.result;

	var macd_above = false;
	var rsi_above = false;
	var cci_above = false;

	var found_treshold = false;

	for (let i = 0; i < this.macd_above.length; i++) {
		if (this.macd_above[i] < 0)
		{
			found_treshold = true;
		}
		else if (found_treshold && this.macd_above[i] > 0)
		{
			macd_above = true;
			break;
		}
	}

	found_treshold = false;

	for (let i = 0; i < this.rsi_above.length; i++) {
		if (this.rsi_above[i] < 30)
		{
			found_treshold = true;
		}
		else if (found_treshold && this.rsi_above[i] > 30)
		{
			rsi_above = true;
			break;
		}
	}

	found_treshold = false;

	for (let i = 0; i < this.cci_above.length; i++) {
		if (this.cci_above[i] < -100)
		{
			found_treshold = true;
		}
		else if (found_treshold && this.cci_above[i] > -100)
		{
			cci_above = true;
			break;
		}
	}

    if (macd_above && rsi_above && cci_above)
    {
      this.advice({
        direction: 'long', // or short
        trigger: { // ignored when direction is not "long"
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
