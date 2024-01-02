import discord
from discord.ext import commands, tasks
import datetime
import os

##########################################
# Initializsation of all the main values #
##########################################
text_file = open("./crypto_id.txt", "r")
data = text_file.read().split('\n')

my_id = data[0]
hsh_channel_id = data[1]

text_file.close()
##########################################

crypto_log_cnt = 0

async def check_log(client):
    text_file = open("/tmp/log/crypto.log", "r")
    data = text_file.read().split('\n')
    data = list(filter(lambda element: "INFO" in element, data))
    text_file.close()
    global crypto_log_cnt

    if (len(data) > 23 and crypto_log_cnt < len(data)):
        crypto_log_cnt = len(data)
    
        original_bal = data[-3:][0].split(' ')[6]
        current_bal = data[-3:][1].split(' ')[6]
        profit_per = data[-3:][2].split(' ')[5]

        time = data[-3:][2].split(' ')[0] + " " + data[-3:][2].split(' ')[1]
        res = "[CRYPTO] - " + time + "\nThe original balance is: " + original_bal + "\nThe current balance is: " + current_bal + "\nThe profit is: " + profit_per + "\n<@" + str(my_id) + ">"

        await client.get_channel(hsh_channel_id).send(res)

async def is_log(msg_channel):
    channel_id = 1190046189463293963

    if (not os.path.exists("/tmp/log/crypto.log")):
        await msg_channel.send("There is no crypto.log file")
    else:
        text_file = open("/tmp/log/crypto.log", "r")
        data = text_file.read().split('\n')
        data = list(filter(lambda element: "INFO" in element, data))
        text_file.close()
        time = time = data[-3:][2].split(' ')[0] + " " + data[-3:][2].split(' ')[1]
        res = "[CRYPTO] - There are actually " + str(len(data)) + " lines in the crypto.log file.\nThe time of the last log is : " + time
        await msg_channel.send(res)