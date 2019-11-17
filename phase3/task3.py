from database import Database
from phase3.ppr import PageRank


def starter():
    image_vectors = Database().retrieve_many()
    
    graph = PageRank(image_vectors)
    

if __name__ == '__main__':
    starter()
