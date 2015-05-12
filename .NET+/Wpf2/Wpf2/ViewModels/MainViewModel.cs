using GalaSoft.MvvmLight;
using System.Collections.ObjectModel;
using Wpf2.Models;
using Wpf2.ViewModels;
using System.Linq;
using System.Windows.Input;
using GalaSoft.MvvmLight.Command;
using System.ComponentModel;


namespace Wpf2.ViewModel
{
    public class MainViewModel : ViewModelBase
    {
        public ICommand JumpCommand { get; set; }

        public ICommand ChangeCommand { get; set; }


        private ObservableCollection<PersonViewModel> persons;

        public ObservableCollection<PersonViewModel> Persons 
        {
            get
            {
           
                return persons;
            }
            private set
            {
                persons = value;
            }
        }

        private PersonViewModel selectedperson;

        public PersonViewModel SelectedPerson
        {
            get { return selectedperson; }
            set {
                Set(() => SelectedPerson, ref selectedperson, value);
                CommandManager.InvalidateRequerySuggested();
                //if (value == selectedperson) return;      négy sor helyett egy
                //selectedperson = value
                //if (PropertyChanged != null) PropertyChanged(new PropertyChangedEventArgs("SelectedPerson")); 
                //RaisePropertyChanged (()=>selectedperson);
                }
        }



        public MainViewModel()
        {
            Persons = new ObservableCollection<PersonViewModel>(
                PersonManager.Persons.Select(p => new PersonViewModel(p)));

            JumpCommand = new RelayCommand(() => SelectedPerson = Persons[0]);

            ChangeCommand = new RelayCommand(
                () => SelectedPerson.Person.Name += "qwe",
                () => SelectedPerson != null && SelectedPerson.Person.Age%2!=1);
        }
    }
}