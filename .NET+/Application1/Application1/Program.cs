using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;


// today C#3.0

namespace Application1
{
    class Program
    {
        static void Main(string[] args)
        {
            //var person = new Person {Age = 15,Name = "Kond"}; //pöpec
            List<Person> list = new List<Person>() 
            {   new Person {Age = 15,Name = "Konkd"},
                new Person {Age = 30,Name = "István"},
                new Person {Age = 25,Name = "Klárk"}
            };
          
            //delegate típus, visszatérési értékkel
            Func<int, int, int> func = ((a, b )=> a * b); //konkrétan: kér 1 intet, és visszatér 1-el
            Console.WriteLine(func(2, 4));
            //expression-ök -> nagyon magas...
            Expression<Func<int, int>> expr =(a=>a*7);  //kifejezésfa
            var func2 = expr.Compile();
            Console.WriteLine(func2(3));


            var queryable = list.AsQueryable();



            Console.ReadKey();
            
        }

        static bool isWiseJedi (Person person) // ez a függvényt jobb lenne inline definiálni, mivel csak egyszer fogjuk használni
        {
            return person.Age > 5;
        }

   
//először ezt csináltuk
        void property()
        {
            Console.WriteLine("hali");

            Person Pali = new Person();
            Pali.AgeChanged += PersonAgeChanged;
            Pali.Age = 10;
            Pali.Age = 1;
        }
        static void PersonAgeChanged(int age)
        {
            Console.WriteLine(age);
        }
////////////////////////
// aztán ez
        void Variable()
        {
            //variable-ök, dynamic, mint a javascript

            //egyszerű típusok esetén ne használjuk a var kulcsszót
            var i = 4;
            var d = 4.0;
            var s = "string";


            // i = "asd" ez már nem megy, de i=10 még menne

            var arr = new[] { 1, 2.1, 3, 4, 5 };

            Console.WriteLine(arr.GetType());
            Console.WriteLine(i.GetType());
            Console.WriteLine(d.GetType());
            Console.WriteLine(s.GetType());

            var dic = new Dictionary<AccessViolationException, List<Person>>(); //ilyenkor nagyon szép a var, és a konstruktorban úgyis látszik, hogy mi van

            //ha akarjuk tudni inicializálva volt-e
            int? ager = 10; //sokkal szebbszintén nullable //Nullable<int> ager =10;  

            ager = null;         // HasValue billeg ahogy kell
            // if (!ager.HasValue)
            Console.WriteLine(ager == null ? 3 : ager.Value); //ha csekkelni akarunk, egyébként default
            /// még szebben
            Console.WriteLine(ager ?? 3);
        }
////////////////////////////////////////
        void others ()
        {
            //var person = new Person {Age = 15,Name = "Kond"}; //pöpec
            List<Person> list = new List<Person>() 
            {   new Person {Age = 15,Name = "Konkd"},
                new Person {Age = 30,Name = "István"},
                new Person {Age = 25,Name = "Klárk"}
            };
          
            /*
          foreach (var item in PersonHelper.Filter(list, "k"))
          {
              Console.WriteLine(item.ToString());
               
          }*/
            // bővítőmetódusos
            foreach (var item in list.Filter("s"))
            {
                Console.WriteLine(item.ToString());

            }
            var tomb = new int[] { 1, 5, 1100, 6 };
            Console.WriteLine(tomb.Max<int>().ToString());
            foreach (var item in list.Where(    ///IsWiseJedi))     de inline pöpecebb
                //delegate (Person person) { return person.Age > 5; }
                 person => person.Age > 5// méég egyszerűbben                   ||| kapott érték => amivel visszatér
                ))
            {
                Console.WriteLine(item);


            }
            Console.WriteLine(list.Average(p => p.Age));
            Console.WriteLine(list.Any(p => p.Name.Contains("a")));

            foreach (var item in list.OrderByDescending(p => p.Name))
            {
                Console.WriteLine(item);
            }

            foreach (var item in list
                .Where(p => p.Age > 5) //függvényláncolás
                .Where(p => p.Name.Contains("y")))
            {

            }

            var a1 = new { Name = "Béla", Age = 5 }; //létrehozott új osztályt, de elég gagyi
            var a2 = new { Name = "Béla2", Age = 6 };
            Console.WriteLine(a1); //automatikus tostring, okos
            Console.WriteLine(a1.GetType());




            /////////////////////////////////////////////////
            var query = list
                .Where(p => p.Age > 5)
                .OrderByDescending(p => p.Name)
                .Select(p => "hello " + p.Name + ".");


            foreach (var item in query)
            {
                Console.WriteLine(item);
            }

            //itt kezdődik a linq -> Language Integrated Query
            // a kettő kód teljesen ekvivalens
            var query2 =
                from p in list
                where p.Age > 5
                orderby p.Name descending
                // select "ohájóó " + p.Name + ".";
                select new
                {
                    Name = p.Name,
                    BirthYear = DateTime.Now.AddYears(-p.Age)
                };
            foreach (var item in query2)
            {
                Console.WriteLine(item);
            }

            var query3 = query.Where(p => p.Contains("Y"));

            
        }
    }




    public class Person
    {
        public delegate void AgeChangedHandler(int newAge);
        
        //prop+TAB+TAB

        public string Name { get; set; }

        //propf+TAB+TAB

        //past tense, mivel változás után
        public event AgeChangedHandler AgeChanged; //event nélkül máshonnan is lehet hívni, + lehet = a += helyett is, ami ROSSZ, felülírja a feliratkozottakat
        
        private int age;

        
        public int Age
        {
            get { return age; }
            set {
                if (age == value) return;
                age = value;
                if (AgeChanged != null)
                    AgeChanged(age);
                }
        }

        public override string ToString()
        {
            return (Name+ "("+ Age.ToString()+ ")");
        }

    }
}
