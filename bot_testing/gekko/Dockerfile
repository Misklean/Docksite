FROM node:alpine

RUN apk add --no-cache python3 make g++ git
ENV PATH="/usr/local/bin:$PATH"

# Create app directory
WORKDIR /gekko

# Install GYP dependencies globally, will be used to code build other dependencies
RUN npm install -g --production node-gyp && \
    npm cache clean --force

# Install Gekko dependencies
COPY package.json .
RUN npm install --production && \
    npm install --production talib tulip pg && \
    npm cache clean --force

# Install Gekko Broker dependencies
WORKDIR exchange
COPY exchange/package.json .
RUN npm install --production && \
    npm cache clean --force
WORKDIR ../

# Bundle app source
COPY . .

CMD node gekko --config myconfig.js > /tmp/log/crypto.log