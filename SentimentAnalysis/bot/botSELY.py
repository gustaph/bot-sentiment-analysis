import tweepy
import pandas as pd
import numpy as np
import re
import string
from time import sleep
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def createAPI():
    consumerKey = ''
    consumerSecret = ''
    accessToken = ''
    accessSecret = ''

    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
    authenticate.set_access_token(accessToken, accessSecret)
    api = tweepy.API(authenticate, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
    except Exception as e:
        logger.error('Error creating API', exc_info=True)
        raise e
    logger.info('API created')
    return api

def getPosts(api, userName, countLimit, mention):
    try:
        tweets = api.user_timeline(screen_name=userName, count=countLimit, lang='pt', tweet_mode='extended')
        df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['texto'])
        df['texto'] = df['texto'].apply(cleanText)
        df.to_csv('dados.csv', encoding='utf-8', index=False)
    except Exception as e:
        api.update_status(f'@{mention.user.screen_name} Oi, {mention.user.screen_name}, beleza? Dei uma olhadinha aqui, e percebi que voçê ainda não possui {countLimit} tweets.\n❗- Voçê pode tentar de novo, mas dessa vez com um número de tweets válidos ok? 👀', mention.id)
        logger.error('Unable to retrieve tweets')
        
def cleanText(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def retriveLastSeenID(fileName):
    f_read = open(fileName, 'r')
    lastSeenID = int(f_read.read().strip())
    f_read.close()
    return lastSeenID

def storeLastSeenID(lastSeenID, fileName):
    f_write = open(fileName, 'w')
    f_write.write(str(lastSeenID))
    f_write.close()
    return

def checkMentions(api):
    MAX = 1000

    logger.info('Retrieving mentions')
    lastSeenID = retriveLastSeenID('last_seen_id.txt')
    mentions = api.mentions_timeline(lastSeenID, tweet_mode='extended')

    for mention in reversed(mentions):
        lastSeenID = mention.id
        storeLastSeenID(lastSeenID, 'last_seen_id.txt')

        newSentence = mention.full_text.split()
        count = ' '
        if 'help' in newSentence[1]:
           api.update_status(f'@{mention.user.screen_name} Oi, {mention.user.screen_name}, tudo bem? É muito fácil usar minhas funcionalidades, da uma olhada:\n✔️ - Para analisar um número x de tweets: @\BotSely x (para x menor ou igual a 1000)\n ✔️ - Para analisar 1000 tweets (max): @\BotSely max', mention.id)
        elif 'max' in newSentence[1]:
            getPosts(api, mention.user.screen_name, MAX, mention)
        elif newSentence[1][0] in '1234567890':
            for number in newSentence[1]:
                if number in '1234567890':
                    count = count + number
            if int(count) <= MAX:
                getPosts(api, mention.user.screen_name, int(count), mention)
            else:
                api.update_status(f'@{mention.user.screen_name} Oi, {mention.user.screen_name}, não sei se eu consigo ler tantos tweets assim :(\n⚠️ - Para consultar mais informações: @\BotSely help', mention.id)
        else:
            continue

def main():
    api = createAPI()
    while True:
        checkMentions(api)
        sleep(20)

if __name__ == '__main__':
    main()