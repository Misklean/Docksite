import discord
from discord.ext import commands, tasks
import datetime
import os

from hsh_crypto import check_log, is_log

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

##########################################
# Initializsation of all the main values #
##########################################
text_file = open("./main_id.txt", "r")
data = text_file.read().split('\n')

token = data[0]
my_id = data[1]
hsh_channel_id = data[2]

text_file.close()
##########################################

client = commands.Bot(command_prefix = ".", intents=intents)
# -----------------------------------------------------------------------

@tasks.loop(hours=24)
async def loop_duolinguo():

    time = datetime.datetime.combine(datetime.date.today(), datetime.time(23, 00, 00))
    await discord.utils.sleep_until(time)

    res = "You should do your Duolinguo <@" + str(my_id) + ">!"
    await client.get_channel(hsh_channel_id).send(res)

@tasks.loop(hours=24)
async def loop_sport():

    time = datetime.datetime.combine(datetime.date.today(), datetime.time(22, 00, 00))
    await discord.utils.sleep_until(time)

    res = "You should do your sport <@" + str(my_id) + ">!"
    await client.get_channel(hsh_channel_id).send(res)

@tasks.loop(minutes=1)
async def loop_crypto_log():
    await check_log(client)

async def give_crypto_log(msg_channel):
    await is_log(msg_channel)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('.hello'):
        await message.channel.send('Hello!')

    if message.content.startswith('.crypto'):
        await give_crypto_log(message.channel)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    loop_crypto_log.start()
    loop_duolinguo.start()
    loop_sport.start()

client.run(token)