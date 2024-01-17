"""
 /*************************************************************************
 * 
 * CONFIDENTIAL
 * __________________
 * 
 *  Copyright (2023-2024) AI Labs, IronOne Technologies, LLC
 *  All Rights Reserved
 *
 *  Author  : Theekshana Samaradiwakara
 *  Description :Python Backend API to chat with research paper data  
 *  CreatedDate : 10/01/2024
 *  LastModifiedDate : 10/01/2024
 *************************************************************************/
"""
# import os
# import sys

import logging
import datetime
logging.info(f"-------------------------- 000000000000 Hello  initialize API! logging-----------------------------")

import uvicorn
# from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware

def filer():
    # return "logs/log "
    today = datetime.datetime.today()
    # log_filename = f"logs/{today.year}-{today.month:02d}-{today.day:02d}.log"
    log_filename = f"logs/app.log"
    return log_filename

file_handler = logging.FileHandler(filer())
file_handler = logging.handlers.TimedRotatingFileHandler(filer(),when="D")
file_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s (%(name)s) : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler],
    force=True,
)

logger = logging.getLogger(__name__)

# load_dotenv()
# host= os.environ.get('APP_HOST')
# port= int(os.environ.get('APP_PORT'))


class ChatAPI:

    def __init__(self): 
        self.router = APIRouter()

        # self.router.add_api_route("/", self.hello, methods=["GET"])
        self.router.add_api_route("/hello", self.hello, methods=["GET"])
        self.router.add_api_route("/health", self.hello, methods=["GET"])

        self.router.add_api_route("/chat", self.chat, methods=["GET", "POST"])

    async def hello(self):
        logger.info(f"-------------------------- Hello there! -----------------------------")
        logging.info(f"-------------------------- Hello there! -----------------------------")
        return "Hello there!"
        
    
    async def chat(self, userQuery):# -> ResponseModel: #chat: QueryModel): # -> ResponseModel:
        """Makes query to doc store via Langchain pipeline.

        :param chat.: question, model, dataset location, history of the chat.
        :type chat: QueryModel
        """
        logger.info(f"-------------------------- userQuery: {userQuery} -------------------------- ")

        try:
            res = 'get_QA_Answers(userQuery)'
            logger.info(f"--------------------------  answer: {res} -------------------------- ")
            # return res
            return res
        
        except HTTPException as e:
            logger.exception(e)
            raise e
                    
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=400, detail=f'Error : {e}')
   

# initialize API
logger.info(f"-------------------------- Hello  initialize API! logger -----------------------------")
logging.info(f"-------------------------- Hello  initialize API! logging-----------------------------")
app = FastAPI(title="Library chatbot API")
api = ChatAPI()
app.include_router(api.router)

# origins = ['http://localhost:8000','http://192.168.10.100:8000']

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# if __name__ == "__main__":
#     host = '0.0.0.0'
#     port = 8080
    
#     # config = uvicorn.Config("server:app",host=host, port=port, log_config= logging.basicConfig())
#     config = uvicorn.Config("server:app",host=host, port=port)
#     server = uvicorn.Server(config)
#     server.run()

#     # uvicorn.run(app)