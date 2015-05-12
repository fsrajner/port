using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Application1
{
    public static class PersonHelper
    {
       // public static IEnumerable<Person> Filter(IEnumerable<Person> list, string filter)
        
        public static IEnumerable<Person> Filter(this IEnumerable<Person> list, string filter) //bővítómetódus
            //List<Person> Filter(List<Person> list, string filter) 
        {
            //var result = new List<Person>();
            foreach (var item in list)
            {
                if (item.Name.Contains(filter))
                    yield return item;           // így nincs a plusz value, és átláthatóbb is
                   // result.Add(item);             plusz! két szál együtt máködik, és egymásnak adogatják
                                                // nem gyárt felesleges szemetet sok elem esetén se
            }
            yield return new Person { Name = "Béla" };
            yield break;                               //átadja az értéket a yield, majd onnan folytatja
            //return result;
        }
    }
}
